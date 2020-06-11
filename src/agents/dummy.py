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


class ClientDummyAgent(Thread):
    """Dummy NN basing on the naive agent"""

    def __init__(self, start_level=1):
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

        # Initialize the dummy predictor NN
        self.predictor = keras.Sequential(
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
                tf.keras.layers.Dense(32, activation='relu'),
                tf.keras.layers.Dense(3, activation='sigmoid')
            ]
        )

        print('Dummy NN agent initialized.')

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

        # Initialization
        self.ar.configure(self.id)
        self.observer_ar.configure(self.id)
        n_levels = self.update_no_of_levels()
        self.solved = n_levels * [0]

        # Check the current score
        self.check_my_score()

        # Load the level in Science Birds
        self.ar.load_level(self.current_level)

        # Fork thread for parallel, periodical screenshot updates
        # gt_thread = Thread(target=self.sample_state)
        # gt_thread.start()

        # Until stop by user, do one loop iteration per level
        while True:
            print('solving level: %d' % self.current_level)

            # Try to solve the level and obtain resulting game status
            state = self.solve()

            # If the level is solved, go to the next level
            if state == GameState.WON:

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
            elif state == GameState.LOST:

                # Check for change of number of levels in the game
                self.update_no_of_levels()

                # Check the current score
                self.check_my_score()

                # Increase the failed counter for this level
                self.failed_counter += 1

                # Restart the level
                self.ar.restart_level()

            # Handle unexpected cases
            elif state == GameState.LEVEL_SELECTION:
                print("Unexpected level selection page, go to the last current level: "
                      , self.current_level)
                self.ar.load_level(self.current_level)

            elif state == GameState.MAIN_MENU:
                print("Unexpected main menu page, go to the last current level: "
                      , self.current_level)
                self.ar.load_level(self.current_level)

            elif state == GameState.EPISODE_MENU:
                print("Unexpected episode menu page, go to the last current level: "
                      , self.current_level)
                self.ar.load_level(self.current_level)

    def getScreenshot(self):
        """Fetch the current game screenshot."""

        image, _ = self.ar.get_ground_truth_with_screenshot()
        return image

    def solve(self):
        """
        Solve the current level by shooting birds.
        :return: GameState: the game state after all shots are performed.
        """
        # Sling reference point coordinates
        sling_ref_point_x = 191
        sling_ref_point_y = 344
        # TODO: Does the sling reference point change?

        # Get current Screenshot
        image = self.getScreenshot()

        if image is None:
            print("Got a screenshot which is None.")
            return self.ar.get_game_state()

        # Crop the image to reduce information overload, and plot it.
        # The cropped image has then dimension (325, 800, 3).
        image = image[75:400, 40:]
        plt.imshow(image)
        plt.show()

        # Predict the optimal next shot
        dx, dy, tap_time = self.plan(image)

        # Validate the predicted shot parameters
        # TODO: Verify these constraints
        if dx >= 0:
            print("The agent tries to shoot to the left!")
        if dx < -80:
            print("Sling stretch too strong in x-direction.")
        if dy < -80 or dy > 80:
            print("Sling stretch too strong in y-direction.")
        if tap_time > 4000:
            print("Very long tap time!")

        # dx = int(release_point_x - sling_ref_point_x)
        # dy = int(release_point_y - sling_ref_point_y)

        # Perform the shot
        print("Shooting with dx = %d, dy = %d, tap_time = %d" % (dx, dy, tap_time))
        self.ar.shoot(sling_ref_point_x, sling_ref_point_y, dx, dy, 0, tap_time, isPolar=False)

        # Update the state and return it
        state = self.ar.get_game_state()
        return state

    def plan(self, image):
        """
        Given a screenshot of the game, an NN is used to predict a good shot.
        :return: dx, dy:
        tap_time:
        """

        # Normalize the image
        image = image.astype(np.float32)
        image = image/255
        image = np.reshape(image, (1, 325, 800, 3))

        # Do the prediction
        pred = self.predictor.predict(image)

        # Convert the prediction (denormalize)
        pred = pred[0]
        dx = int(pred[0] * -80)
        dy = int(pred[1] * 160 - 80)
        tap_time = int(pred[2] * 4000)

        return dx, dy, tap_time


if __name__ == "__main__":
    na = ClientDummyAgent()
    na.run()
