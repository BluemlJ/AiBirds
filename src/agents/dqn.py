import time
import matplotlib.pyplot as plt
import json
import socket
import numpy as np
import cv2
from threading import Thread
import tensorflow as tf
from tensorflow import keras

from src.client.agent_client import AgentClient, GameState

num_episodes = 1000
angle_res = 20  # angle resolution: the number of possible (discretized) shot angles
tap_time_res = 10  # tap time resolution: the number of possible tap times
# action_size = 6  # the action space discretization resolution in each dimension
state_x_dim = 124
learning_rate = 0.01
grace_factor = 0.1
score_normalization = 100000
phi = 10
psi = 40
max_t = 4000
sync_period = 100  # Number of levels between each synchronization of online and target network

# Sling reference point coordinates TODO: Do these coordinates change?
sling_ref_point_x = 191
sling_ref_point_y = 344


class ClientDQNAgent(Thread):
    """Deep Q-Network (DQN) agent"""

    def __init__(self, start_level=1, sim_speed=1, optimizer=tf.optimizers.Adam(learning_rate=learning_rate),
                 gamma=0.9):
        super().__init__()

        with open('./src/client/server_client_config.json', 'r') as config:
            sc_json_config = json.load(config)
        self.ar = AgentClient(**sc_json_config[0])

        with open('./src/client/server_observer_client_config.json', 'r') as observer_config:
            observer_sc_json_config = json.load(observer_config)
        self.observer_ar = AgentClient(**observer_sc_json_config[0])

        print("Connecting agent to server...")
        try:
            self.ar.connect_to_server()
        except socket.error as e:
            print("Error in client-server communication: " + str(e))

        print("Connecting observer agent to server...")
        try:
            self.observer_ar.connect_to_server()
        except socket.error as e:
            print("Error in client-server communication: " + str(e))

        # General model parameters
        self.solved = []
        self.id = 28888

        # Game parameters
        self.current_level = start_level
        self.sim_speed = sim_speed

        # Discount factor
        self.gamma = gamma

        # Parameters for cooling down epsilon greedy
        self.epsilon = 1
        self.cool_down = 0.98

        # Training optimizer
        self._optimizer = optimizer

        # Initialize the architecture of the acting part of the DQN, theta
        self.online_network = self._build_compile_model()

        # Initialize the architecture of the learning part of the DQN (identical to above), theta-
        self.target_network = self.online_network

        print('DQN agent initialized.')

    def _build_compile_model(self):
        model = keras.Sequential(
            [
                tf.keras.layers.Conv2D(32, (6, 6), activation='relu', input_shape=(state_x_dim, state_x_dim, 3)),
                tf.keras.layers.MaxPooling2D((2, 2)),
                tf.keras.layers.Dropout(0.25),
                tf.keras.layers.Conv2D(64, (4, 4), activation='relu'),
                tf.keras.layers.MaxPooling2D((2, 2)),
                tf.keras.layers.Dropout(0.5),
                tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
                tf.keras.layers.Dropout(0.5),
                tf.keras.layers.Flatten(),
                tf.keras.layers.Dense(200, activation='relu'),
                tf.keras.layers.Dense(angle_res * tap_time_res, activation='relu')
            ]
        )

        model.compile(loss='mse', optimizer=self._optimizer)

        return model

    def sample_state(self, cycle_time=0.5):
        """
        Sample a screenshot from the observer agent. This method can be run in a different thread.
        NOTE: Use cycle_time of > 0.01, otherwise it may cause a lag in Science Birds game
              due to the calculation of groundtruth data.
        """

        while True:
            vision, _ = self.observer_ar.get_ground_truth_with_screenshot()
            time.sleep(cycle_time)

    def get_next_level(self):
        """Obtain the number of the next level to play."""
        return self.current_level + 1

    def check_my_score(self):
        """Get all level scores and mark levels with positive score as solved."""
        print("My score:")

        scores = self.ar.get_all_level_scores()
        for level, score in enumerate(scores):
            # If score is positive, mark level as solved and print that score
            if score > 0:
                self.solved[level] = 1
                print("Level ", level + 1, ":", score)

        return scores

    def update_no_of_levels(self):
        """
        Checks if the number of levels has changed in the game. (Why do we need that?)
        If yes, the list of solved levels is adjusted to the new number of levels.
        """

        n_levels = self.ar.get_number_of_levels()

        # if number of levels has changed, adjust the list of solved levels to the new number
        if n_levels > len(self.solved):
            for n in range(len(self.solved), n_levels):
                self.solved.append(0)

        if n_levels < len(self.solved):
            self.solved = self.solved[:n_levels]

        print('Number of levels: ' + str(n_levels))

        return n_levels

    def run(self):
        """The agent's main running routine."""
        print("Starting DQN agent...")

        # Initialization
        self.ar.configure(self.id)
        self.observer_ar.configure(self.id)
        self.ar.set_game_simulation_speed(self.sim_speed)

        # Load the first level to train in Science Birds
        self.ar.load_level(self.current_level)

        # Train the network for num_episodes episodes
        print("Starting training...")
        self.train()
        print("Training finished successfully!")

    def train(self):
        # Initialize experience buffer, containing all (s, a, r, s', t) tuples experienced so far
        experience = np.empty((0, 5))

        # Initialize return list (list of final scores for each level played)
        returns = []

        for i in range(num_episodes):
            print("\nEpisode %d, Level %d" % (i + 1, self.ar.get_current_level()))

            # Observations during the current level: list of (s, a, r, s', t) tuples
            obs = []

            # Initialize current episode's return to 0
            ret = 0

            # Initialize variable to monitor the application's state
            appl_state = self.ar.get_game_state()

            # Get the environment state (preprocessed screenshot) and save it
            env_state = self.get_state()

            # Try to solve a level and collect observations (actions, environment state, reward)
            while appl_state == GameState.PLAYING:
                # Plot the current state
                # self.plot_state(env_state)

                # Predict the next action to take, i.e. the best shot
                action = self.plan(env_state)

                # Perform shot, observe new environment state, level score and application state
                next_env_state, score, appl_state = self.shoot(action)

                # Compute reward (to be the number of additional points gained with the last shot)
                reward = score - ret

                # Observe if this level was terminated
                terminal = not appl_state == GameState.PLAYING

                # Save new experience (s, a, r, s', t)
                obs += [(env_state, action, reward, next_env_state, terminal)]

                # Update return
                ret = score

                # Update old env_state with new one
                env_state = next_env_state

            if not (appl_state == GameState.WON or appl_state == GameState.LOST):
                print("Error: unexpected application state. The application state is neither WON nor LOST. "
                      "Skipping this training iteration...")
                break

            # Convert observations list into np.array
            obs = np.array(obs)

            # If the level is lost, update the return. In the actual case, reward and return would
            # be zero, but we give some "grace" points, so the network can learn even if it constantly looses.
            if appl_state == GameState.LOST:
                # Grace points on return
                ret *= grace_factor

                # Grace points on all the rewards given during this level
                obs[:, 2] *= grace_factor

            print("Got return", ret)

            # Save the return
            returns += [ret]

            # Every X episodes, plot the return graph
            if (i + 1) % 10 == 0:
                self.plot_returns(returns)

            # Append observations to experience buffer
            experience = np.append(experience, obs, axis=0)

            # Load next level (in advance)
            self.load_next_level(appl_state)

            # Train every X levels
            if (i + 1) % 10 == 0:
                # Update network weights to fit the experience
                print("Learning from experience...")
                self.update(experience)

            # Synchronize target and online network every Y levels
            if (i + 1) % sync_period == 0:
                self.target_network = self.online_network

            # Cool down: reduce epsilon to reduce randomness
            self.epsilon *= self.cool_down

    def update(self, experience):
        """Updates the network weights. This is the actual learning step of the agent."""

        # Obtain number of experienced transitions
        exp_len = experience.shape[0]

        # Obtain batch size
        batch_size = np.min((exp_len, 32))

        # Select a random batch from experience to train on, TODO: implement Expereince Replay
        batch_ids = np.random.choice(exp_len, batch_size)
        batch = experience[batch_ids]

        states = []
        targets = []

        for state, action, reward, next_state, terminal in batch:

            # Predict Q-value matrix for given state
            target = self.target_network.predict(state)

            # Refine Q-values with observed reward
            if terminal:
                # If this is the last step, take the reward plainly
                target[0][action] = reward
            else:
                # Else, use reward of this step plus discounted expected return
                t = self.target_network.predict(next_state)
                target[0][action] = reward + self.gamma * np.amax(t)

            states += [state[0]]
            targets += [target[0]]

        states = np.asarray(states)
        targets = np.asarray(targets)

        # Update the online network's weights
        self.online_network.fit(states, targets, epochs=1, verbose=0, batch_size=batch_size)

    def plan(self, state):
        """
        Given a state of the game, the deep Q-learner NN is used to predict a good shot.
        :return: dx, dy:
        tap_time:
        """

        # Obtain action-value pairs (Q)
        q_vals = self.online_network.predict(state)

        if np.random.random(1) > self.epsilon:
            # Determine optimal action as usual
            # Extract the action which has highest predicted Q-value
            action = q_vals.argmax()
        else:
            # Choose action by random
            action = np.random.randint(angle_res * tap_time_res)

        print("Expected return:", int(np.amax(q_vals) * score_normalization))

        return action

    def shoot(self, action):
        """Performs a shot and observes and returns the consequences."""

        # Convert action index into aim vector and tap time
        dx, dy, tap_time = self.action_to_params(action)

        '''# TODO: Verify if these constraints are useful
        dx = int(action[0] / action_size * -80)  # in interval [-80, -10]
        dy = int(action[1] / action_size * 160 - 80)  # in interval [-50, ]
        tap_time = int(action[2] / action_size * 3000)  # in interval [0, 3000]

        # Validate the predicted shot parameters
        if dx > 0:
            print("The agent tries to shoot to the left!")
        if dx < -80:
            print("Sling stretch too strong in x-direction.")
        if dy < -80 or dy > 80:
            print("Sling stretch too strong in y-direction.")
        if tap_time > 3000:
            print("Very long tap time!")'''

        # Perform the shot
        print("Shooting with dx = %d, dy = %d, tap_time = %d" % (dx, dy, tap_time))
        self.ar.shoot(sling_ref_point_x, sling_ref_point_y, dx, dy, 0, tap_time, isPolar=False)

        print("Collecting observations...")

        # Get the environment state (cropped screenshot)
        env_state = self.get_state()

        # Obtain in-game score, normalized
        score = self.ar.get_current_score() / score_normalization

        # Get the application state and return it
        appl_state = self.ar.get_game_state()

        return env_state, score, appl_state

    def action_to_params(self, action):
        """Converts a given action index into the corresponding angle and tap time."""

        # Convert the action index into index pair, indicating angle and tap_time
        action = np.unravel_index(action, (angle_res, tap_time_res))
        print("Action:", action)

        # Formula parameters, TODO: to be tuned
        c = 3.6
        d = 1.8

        # Retrieve shot angle alpha
        k = action[0] / angle_res
        alpha = ((1 + 0.5 * c) * k - 3 / 2 * c * k ** 2 + c * k ** 3) ** d * (180 - phi - psi)

        print("alpha: %d" % alpha)

        # Convert angle into vector
        dx, dy = self.angle_to_vector(alpha)

        # Retrieve tap time
        t = action[1]
        tap_time = int(t / 10 * max_t)

        return dx, dy, tap_time

    def angle_to_vector(self, alpha):
        rad_shot_angle = np.deg2rad(alpha + phi)

        dx = - np.sin(rad_shot_angle) * 80
        dy = np.cos(rad_shot_angle) * 80

        return int(dx), int(dy)

    def get_state(self):
        """Fetches the current game screenshot and turns it into a cropped, scaled and normalized pixel matrix."""

        # Obtain game screenshot
        screenshot, _ = self.ar.get_ground_truth_with_screenshot()

        # Crop the image to reduce information overload.
        # The cropped image has then dimension (325, 800, 3).
        crop = screenshot[75:400, 40:]

        # Rescale the image into a (smaller) square
        scaled = cv2.resize(crop, (state_x_dim, state_x_dim))

        # Normalize the scaled and cropped image
        state = np.expand_dims(scaled.astype(np.float32) / 255, axis=0)

        return state

    def plot_state(self, state):
        # De-normalize state into image
        state = np.reshape(state, (325, 800, 3))
        image = (state * 255).astype(np.int)

        # Plot it
        plt.imshow(image)
        plt.show()

    def plot_returns(self, returns):
        returns = np.array(returns) * score_normalization
        plt.plot(returns, label="Raw scores")

        if len(returns) > 10:
            mov_avg_ret = np.cumsum(returns, dtype=float)
            mov_avg_ret[10:] = mov_avg_ret[10:] - mov_avg_ret[:-10]
            mov_avg_ret = mov_avg_ret[10 - 1:] / 10
            mov_avg_ret = np.insert(mov_avg_ret, 0, 5 * [None])
            plt.plot(mov_avg_ret, label="Moving average 10")

        if len(returns) > 100:
            mov_avg_ret = np.cumsum(returns, dtype=float)
            mov_avg_ret[100:] = mov_avg_ret[100:] - mov_avg_ret[:-100]
            mov_avg_ret = mov_avg_ret[100 - 1:] / 100
            mov_avg_ret = np.insert(mov_avg_ret, 0, 50 * [None])
            plt.plot(mov_avg_ret, label="Moving average 100")

        plt.title("Scores gathered so far")
        plt.xlabel("Episode")
        plt.ylabel("Score")
        plt.legend()
        plt.show()

    def load_next_level(self, appl_state):
        """Update the current_level variable and load the next level according to given application state."""

        # In any case, pick a random level between 1 and 200
        next_level = np.random.randint(199) + 1

        self.current_level = next_level

        self.ar.load_level(next_level)


if __name__ == "__main__":
    agent = ClientDQNAgent()
    # agent.q_network.summary()
    agent.run()
