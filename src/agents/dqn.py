import json
import socket
import pickle
import bz2

import cv2
from threading import Thread
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.initializers import VarianceScaling
from src.client.agent_client import AgentClient, GameState
from src.utils.utils import *
from src.utils.Vision import Vision


class ClientDQNAgent(Thread):
    """Deep Q-Network (DQN) agent for playing Angry Birds"""

    def __init__(self, start_level=1, num_episodes=100000, sim_speed=1, learning_rate=0.0001, replay_period=10,
                 sync_period=200, gamma=0.9, epsilon=1, anneal=0.9999, minibatch=32):
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

        # General
        self.id = 28888

        # Game parameters
        self.current_level = start_level
        self.sim_speed = sim_speed  # Simulation speed for Science Birds (max. 50)

        # Vision for obtaining sling reference point
        self.vision = Vision()

        # Action space parameters
        self.angle_res = 20  # angle resolution: the number of possible (discretized) shot angles
        self.tap_time_res = 10  # tap time resolution: the number of possible tap times
        self.max_t = 4000  # maximum tap time (in ms)
        self.phi = 10  # dead shot angle bottom
        self.psi = 40  # dead shot angle top

        # State space resolution (per dimension)
        self.state_res_per_dim = 124

        # Discount factor
        self.gamma = gamma

        # Parameters for annealing epsilon greedy policy
        self.epsilon = epsilon
        self.anneal = anneal

        # Training optimizer and parameters
        self._optimizer = tf.optimizers.Adam(learning_rate=learning_rate)
        self.num_episodes = num_episodes
        self.replay_period = replay_period  # number of levels between each training of the online network
        self.sync_period = sync_period  # number of levels between each synchronization of online and target network
        self.grace_factor = 1  # reward function modifier, giving X % points on failed levels
        self.score_normalization = 10000
        self.minibatch = minibatch

        # Initialize the architecture of the acting part of the DQN, theta
        self.online_network = self._build_compile_model()

        # Initialize the architecture of the learning part of the DQN (identical to above), theta-
        self.target_network = self.online_network

        # Export the experiences to a file
        self.experience_path = "./experiences.bz2"
        self.current_episode_length = 0

        print('DQN agent initialized.')

    def _build_compile_model(self):
        model = keras.Sequential(
            [
                tf.keras.layers.Conv2D(32, (8, 8), strides=4, kernel_initializer=VarianceScaling(scale=2.),
                                       activation='relu', use_bias=False, input_shape=(self.state_res_per_dim, self.state_res_per_dim, 3)),
                # tf.keras.layers.Dropout(0.25),

                tf.keras.layers.Conv2D(64, (4, 4), strides=2, kernel_initializer=VarianceScaling(scale=2.),
                                       activation='relu', use_bias=False),
                # tf.keras.layers.Dropout(0.5),

                tf.keras.layers.Conv2D(64, (3, 3), strides=1, kernel_initializer=VarianceScaling(scale=2.),
                                       activation='relu', use_bias=False),
                # tf.keras.layers.Dropout(0.5),

                tf.keras.layers.Conv2D(1024, (7, 7), strides=1, kernel_initializer=VarianceScaling(scale=2.),
                                       activation='relu', use_bias=False),
                # tf.keras.layers.Dropout(0.5),
                tf.keras.layers.Flatten(),

                tf.keras.layers.Dense(100, activation='relu'),

                tf.keras.layers.Dense(self.angle_res * self.tap_time_res, activation='linear')
            ]
        )

        model.compile(loss='huber_loss', optimizer=self._optimizer)

        return model

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
        self.play()
        print("Training finished successfully!")

    def play(self):
        # Initialize experience buffer, containing all (s, a, r, s', t) tuples experienced so far
        experience = np.empty((0, 5))

        # Initialize priority list, each transition in experience has a corresponding priority
        # for a more effective experience replay during learning
        priorities = []

        # Initialize return list (list of final scores for each level played)
        returns = []

        for i in range(self.num_episodes):
            print("\nEpisode %d, Level %d, epsilon = %f" % (i + 1, self.ar.get_current_level(), self.epsilon))


            # Observations during the current level: list of (s, a, r, s', t) tuples
            obs = []

            # Initialize current episode's return to 0
            ret = 0

            # Initialize current episode length

            # Initialize variable to monitor the application's state
            appl_state = self.ar.get_game_state()

            # Get the environment state (preprocessed screenshot) and save it
            env_state = self.get_state()

            # Try to solve a level and collect observations (actions, environment state, reward)
            while appl_state == GameState.PLAYING:
                # Plot the current state
                # self.plot_state(env_state)

                # Predict the next action to take, i.e. the best shot, and get estimated value
                action, val_estimate = self.plan(env_state)

                # Perform shot, observe new environment state, level score and application state
                next_env_state, score, appl_state = self.shoot(action)

                # Compute reward (to be the number of additional points gained with the last shot)
                reward = score - ret

                # Compute TD error (difference between expected and observed value)
                td_err = score - val_estimate

                # Observe if this level was terminated
                terminal = not appl_state == GameState.PLAYING

                # Get and save priority for this transition
                priorities += [np.max(priorities, initial=1)]

                # Save experienced transition (s, a, r, s', t)
                obs += [(env_state, action, reward, next_env_state, terminal)]
                # update length of current episode
                self.current_episode_length += 1

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
                ret *= self.grace_factor

                # Grace points on all the rewards given during this level
                obs[:, 2] *= self.grace_factor

            print("Got level score %d" % (ret * self.score_normalization))

            # Save the return
            returns += [ret]

            # Every X episodes, plot the score graph
            if (i + 1) % 200 == 0:
                plot_scores(np.array(returns) * self.score_normalization)

            # Append observations to experience buffer
            experience = np.append(experience, obs, axis=0)

            # Load next level (in advance)
            self.load_next_level(appl_state)

            # Train every train_period levels
            if (i + 1) % self.replay_period == 0:
                # Update network weights to fit the experience
                print("\nLearning from experience...")
                self.learn(experience, priorities)

            # Synchronize target and online network every sync_period levels
            if (i + 1) % self.sync_period == 0:
                self.target_network = self.online_network

            # Cool down: reduce epsilon to reduce randomness
            self.epsilon *= self.anneal

    def learn(self, experience, priorities):
        """Updates the online network's weights. This is the actual learning step of the agent."""

        # Obtain number of experienced transitions
        exp_len = experience.shape[0]

        self.export_experience(experience)

        # Obtain batch size
        batch_size = np.min((exp_len, self.minibatch))

        for j in range(self.minibatch):
            # TODO: Sample transition depending on priorities
            break

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
        :return: action, consisting of an index, corresponding to some shot parameters
        """

        # Obtain action-value pairs, Q(s,a)
        q_vals = self.online_network.predict(state)

        if np.random.random(1) > self.epsilon:
            # Determine optimal action as usual
            # Extract the action which has highest predicted Q-value
            action = q_vals.argmax()
        else:
            # Choose action by random
            action = np.random.randint(self.angle_res * self.tap_time_res)

        # Estimate the expected value for this level
        val_estimate = np.amax(q_vals)

        print("Expected level score:", int(val_estimate * self.score_normalization))

        return action, val_estimate

    def shoot(self, action):
        """Performs a shot and observes and returns the consequences."""
        # Get sling reference point coordinates
        sling_x, sling_y = self.vision.get_sling_reference()

        # Convert action index into aim vector and tap time
        dx, dy, tap_time = self.action_to_params(action)

        # Perform the shot
        # print("Shooting with dx = %d, dy = %d, tap_time = %d" % (dx, dy, tap_time))
        self.ar.shoot(sling_x, sling_y, dx, dy, 0, tap_time, isPolar=False)

        # Get the environment state (cropped screenshot)
        env_state = self.get_state()

        # Obtain in-game score, normalized
        score = self.ar.get_current_score() / self.score_normalization

        # Get the application state and return it
        appl_state = self.ar.get_game_state()

        return env_state, score, appl_state

    def action_to_params(self, action):
        """Converts a given action index into corresponding dx, dy, and tap time."""

        # Convert the action index into index pair, indicating angle and tap_time
        action = np.unravel_index(action, (self.angle_res, self.tap_time_res))

        # Formula parameters, TODO: to be tuned
        c = 3.6
        d = 1.3

        # Retrieve shot angle alpha
        k = action[0] / self.angle_res
        alpha = ((1 + 0.5 * c) * k - 3 / 2 * c * k ** 2 + c * k ** 3) ** d * (180 - self.phi - self.psi) + self.phi

        # Convert angle into vector
        dx, dy = angle_to_vector(alpha)

        # Retrieve tap time
        t = action[1]
        tap_time = int(t / 10 * self.max_t)

        print("Shooting with: alpha = %d °, tap time = %d ms" % (alpha, tap_time))

        return dx, dy, tap_time

    def get_state(self):
        """Fetches the current game screenshot and turns it into a cropped, scaled and normalized pixel matrix."""

        # Obtain game screenshot
        screenshot, ground_truth = self.ar.get_ground_truth_with_screenshot()

        # Update Vision (to get an up-to-date sling reference point)
        self.vision.update(screenshot, ground_truth)

        # Crop the image to reduce information overload.
        # The cropped image has then dimension (325, 800, 3).
        crop = screenshot[75:400, 40:]

        # Rescale the image into a (smaller) square
        scaled = cv2.resize(crop, (self.state_res_per_dim, self.state_res_per_dim))

        # Normalize the scaled and cropped image
        state = np.expand_dims(scaled.astype(np.float32) / 255, axis=0)

        return state

    def load_next_level(self, appl_state):
        """Update the current_level variable and load the next level according to given application state."""

        # In any case, pick a random level between 1 and 200
        next_level = np.random.randint(199) + 1

        self.current_level = next_level

        self.ar.load_level(next_level)

    def export_experience(self, experience):
        """Exports the given experience data to a json file. For each level, the sequences of shots is saved.
        Each shot consists of the initial state, the chosen action and the achieved reward."""
        levels = []
        current_level = []
        num_shots = 0

        for i in range(self.current_episode_length):
            # i[0]: img i[1]: action i[2]: reward i[3]: img i[4]: termination
            current_offset = -self.current_episode_length + i
            current_level.append(experience[current_offset][0])
            current_level.append(experience[current_offset][1])
            current_level.append(experience[current_offset][2])
            num_shots += 1
            if experience[current_offset][4]:
                levels.append(current_level)
                current_level = []
                num_shots = 0

        # try to open the file with previous experiences, if not possible: create an empty experience list
        try:
            with bz2.open(self.experience_path, "rb") as f:
                pre_levels = pickle.load(f)
                print("loaded:", len(pre_levels), "levels")
        except EOFError:
            pre_levels = []
        except FileNotFoundError:
            pre_levels = []
        pre_levels.extend(levels)
        print("stored:", len(pre_levels), "levels")
        with bz2.open(self.experience_path, "wb") as f:
            pickle.dump(pre_levels, f)

        # reset the current episode_length
        self.current_episode_length = 0

    def load_experience(self):
        """Load the experience data from the compressed file and return it as a list of transitions.
        Each transition consists of: initial state, action reward, next state, termination."""
        with bz2.open(self.experience_path, "rb") as f:
            levels = pickle.load(f)
            print("loaded:", len(levels), "levels")

        experience = np.empty((0, 5))

        for l in levels:
            for i in range(int(len(l) / 3)):
                # add state, action, reward
                obs = [l[i * 3], l[i * 3 + 1], l[i * 3 + 2]]
                # check if the current transition is not the last transition
                if (i * 3 + 3) < len(l):
                    obs.append(l[i * 3 + 3])
                    obs.append(False)
                else:
                    obs.append(None)
                    obs.append(True)

                # add the current observation to the total list
                obs = np.array([obs])

                experience = np.append(experience, obs, axis=0)

        return experience

if __name__ == "__main__":
    agent = ClientDQNAgent()
    # agent.q_network.summary()
    agent.run()
