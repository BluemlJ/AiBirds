import time
import matplotlib.pyplot as plt
import json
import socket
import logging
import numpy as np
from threading import Thread
import tensorflow as tf
from tensorflow import keras

from src.client.agent_client import AgentClient, GameState
from src.trajectory_planner.trajectory_planner import SimpleTrajectoryPlanner

num_episodes = 1000
action_size = 10  # the action space discretization resolution in each dimension
learning_rate = 0.01

# Sling reference point coordinates
sling_ref_point_x = 191
sling_ref_point_y = 344
# TODO: Does the sling reference point change?


class ClientDQNAgent(Thread):
    """Deep Q-Nework (DQN) agent"""

    def __init__(self, start_level=1, optimizer=tf.optimizers.Adam(learning_rate=learning_rate), gamma=0.9):
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

        self.current_level = start_level
        self.failed_counter = 0
        self.solved = []
        self.tp = SimpleTrajectoryPlanner()
        self.id = 28888
        self.first_shot = True
        self.prev_target = None
        self._logger = logging.getLogger("ClientNaiveAgent")

        self._optimizer = optimizer

        # Initialize the deep Q-network architecture
        self.q_network = self._build_compile_model()

        # Discount factor
        self.gamma = gamma

        print('DQN agent initialized.')

    def _build_compile_model(self):
        model = keras.Sequential(
            [
                tf.keras.layers.Conv2D(32, (6, 6), activation='relu', input_shape=(325, 800, 3)),
                tf.keras.layers.MaxPooling2D((2, 2)),
                tf.keras.layers.Dropout(0.25),
                tf.keras.layers.Conv2D(64, (4, 4), activation='relu'),
                tf.keras.layers.MaxPooling2D((2, 2)),
                tf.keras.layers.Dropout(0.5),
                tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
                tf.keras.layers.Dropout(0.5),
                tf.keras.layers.Flatten(),
                tf.keras.layers.Dense(200, activation='relu'),
                tf.keras.layers.Dense(action_size**3, activation='sigmoid')
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
                print("Level ", level+1, ":", score)

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
        # n_levels = self.update_no_of_levels()
        # self.solved = n_levels * [0]

        # Check the current score
        # self.check_my_score()

        # Load the first level to train in Science Birds
        self.ar.load_level(self.current_level)

        # Train the network for num_episodes episodes
        print("Starting training...")
        self.train()

        print("Training finished successfully!")

    def train(self):
        # Initialize variable to monitor the application's state
        appl_state = self.ar.get_game_state()

        # Initialize return list (list of final scores for each played level)
        returns = []

        for i in range(num_episodes):
            print("\nEpisode %d" % i)

            # Initialize observation buffer (saves states, actions and rewards)
            obs = []

            # Initialize current episode's return to 0
            ret = 0

            # Get the environment state (cropped screenshot) and save it
            env_state = self.getScreenshot()
            obs += [env_state]

            # Try to solve a level and collect observations (actions, environment state, reward)
            while appl_state == GameState.PLAYING:

                # Predict the next action to take, i.e. the best shot
                action = self.plan(env_state)

                # Perform shot, observe new environment state and level score
                env_state, score, appl_state = self.shoot(action)

                # Compute reward (to be the number of in-game points gained with the last shot)
                reward = score - ret

                # Save observation [action, reward, environment state]
                obs += [action, reward, env_state]

                # Update return
                ret = score

            if not (appl_state == GameState.WON or appl_state == GameState.LOST):
                print("Error: unexpected application state. The application state is neither WON nor LOST. "
                      "Skipping this training iteration...")
                break

            # Save the return
            returns += [ret]

            # Update network weights to fit the newly obtained observations
            self.update(obs)

            # Start the next level
            self.next_level(appl_state)

        return

    def update(self, observations):
        """Updates the network weights. This is the actual learning step of the agent."""

        # Obtain episode length
        ep_len = int((len(observations) - 1)/3)

        for i in range(ep_len):
            state, action, reward, next_state = observations[i*3], observations[i*3+1], observations[i*3+2], observations[i*3+3]

            # Predict Q-values for given state
            target = self.q_network.predict(state)

            # Refine Q-values with observed reward
            if i+4 == len(observations):
                target[0][action] = reward
            else:
                t = self.q_network.predict(next_state)
                target[0][action] = reward + self.gamma * np.amax(t)

            # Update the network weights
            self.q_network.fit(state, target, epochs=1, verbose=0)
        return

    def plan(self, image):
        """
        Given a screenshot of the game, the deep Q-learner NN is used to predict a good shot.
        :return: dx, dy:
        tap_time:
        """

        # Obtain action-value pairs (Q)
        q_vals = self.q_network.predict(image)

        # Reshape the output accordingly
        q_vals = np.reshape(q_vals, (action_size, action_size, action_size))

        # Extract the action which has highest predicted Q-value
        action = q_vals.argmax()

        return action

    def shoot(self, action):
        """Performs a shot and observes and returns the consequences."""

        # Convert it into coordinates and time for environment input
        action = np.unravel_index(action, (action_size, action_size, action_size))
        print("Action:", action)

        dx = int(action[0] / action_size * -80)
        dy = int(action[1] / action_size * 160 - 80)
        tap_time = int(action[2] / action_size * 4000)

        # Validate the predicted shot parameters
        # TODO: Verify these constraints
        if dx > 0:
            print("The agent tries to shoot to the left!")
        if dx < -80:
            print("Sling stretch too strong in x-direction.")
        if dy < -80 or dy > 80:
            print("Sling stretch too strong in y-direction.")
        if tap_time > 4000:
            print("Very long tap time!")

        # Perform the shot
        print("Shooting with dx = %d, dy = %d, tap_time = %d" % (dx, dy, tap_time))
        self.ar.shoot(sling_ref_point_x, sling_ref_point_y, dx, dy, 0, tap_time, isPolar=False)

        print("Collecting observations...")

        # Get the environment state (cropped screenshot)
        env_state = self.getScreenshot()

        # Obtain in-game score
        score = self.ar.get_current_score()

        # Get the application state and return it
        appl_state = self.ar.get_game_state()

        return env_state, score, appl_state

    def getScreenshot(self):
        """Fetch the current game screenshot."""

        image, _ = self.ar.get_ground_truth_with_screenshot()

        # Crop the image to reduce information overload.
        # The cropped image has then dimension (325, 800, 3).
        image = image[75:400, 40:]

        plt.imshow(image)
        plt.show()

        # Normalize the image
        image = image.astype(np.float32)
        image = image/255
        image = np.reshape(image, (1, 325, 800, 3))

        return image

    def next_level(self, appl_state):
        """Update the current_level variable according to given application state."""

        # If the level is solved, go to the next level
        if appl_state == GameState.WON:

            # Check for change of number of levels in the game
            self.update_no_of_levels()

            # Check the current score
            self.check_my_score()

            # Update the level
            self.current_level = self.get_next_level()

            # Load the new level and re-initialize the trajectory planner
            self.ar.load_level(self.current_level)
            self.tp = SimpleTrajectoryPlanner()

        # If lost, then restart the level
        elif appl_state == GameState.LOST:

            # Check for change of number of levels in the game
            self.update_no_of_levels()

            # Check the current score
            self.check_my_score()

            # Increase the failed counter for this level
            self.failed_counter += 1

            # Restart the level
            self.ar.restart_level()

        # Handle unexpected cases
        elif appl_state == GameState.LEVEL_SELECTION:
            print("Unexpected level selection page, go to the last current level: "
                  , self.current_level)
            self.ar.load_level(self.current_level)

        elif appl_state == GameState.MAIN_MENU:
            print("Unexpected main menu page, go to the last current level: "
                  , self.current_level)
            self.ar.load_level(self.current_level)

        elif appl_state == GameState.EPISODE_MENU:
            print("Unexpected episode menu page, go to the last current level: "
                  , self.current_level)
            self.ar.load_level(self.current_level)


if __name__ == "__main__":
    agent = ClientDQNAgent()
    # agent.q_network.summary()
    agent.run()
