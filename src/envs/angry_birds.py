import subprocess
import socket
import cv2
import os
import psutil
import atexit
import numpy as np
import matplotlib.pyplot as plt

from src.envs.env import ParallelEnvironment
from src.envs.ab.agent_client import AgentClient, GameState

# State space
STATE_PIXEL_RES = 128  # width and height of (preprocessed) states

# Action space
ANGLE_RESOLUTION = 20  # the number of possible (discretized) shot angles
TAP_TIME_RESOLUTION = 10  # the number of possible tap times
MAXIMUM_TAP_TIME = 4000  # maximum tap time (in ms)
PHI = 10  # dead shot angle bottom (in degrees)
PSI = 40  # dead shot angle top (in degrees)
ACTIONS = []

SERVER_CLIENT_CONFIG = {
    "requestbufbytes": 4,
    "d": 4,
    "e": 5
}

# Reward
SCORE_NORMALIZATION = 10000

TOTAL_LEVEL_NUMBER = 1300  # non-novelty levels
LIST_OF_VALIDATION_LEVELS = []


def angle_to_vector(alpha):
    rad_shot_angle = np.deg2rad(alpha)

    dx = - np.sin(rad_shot_angle) * 80
    dy = np.cos(rad_shot_angle) * 80

    return int(dx), int(dy)


def action_to_params(action):
    """Converts a given action index into corresponding shot angle and tap time."""

    # Convert the action index into index pair, indicating angle and tap_time
    action = np.unravel_index(action, (ANGLE_RESOLUTION, TAP_TIME_RESOLUTION))

    # Formula parameters
    c = 3.6
    d = 1.3

    # Retrieve shot angle alpha
    k = action[0] / ANGLE_RESOLUTION
    alpha = int(((1 + 0.5 * c) * k - 3 / 2 * c * k ** 2 + c * k ** 3) ** d * (180 - PHI - PSI) + PHI)

    # Retrieve tap time
    t = action[1]
    tap_time = int(t / TAP_TIME_RESOLUTION * MAXIMUM_TAP_TIME)

    return alpha, tap_time


for i in range(ANGLE_RESOLUTION * TAP_TIME_RESOLUTION):
    alpha, tap_time = action_to_params(i)
    ACTIONS += ["alpha = %.1f °, tap_time = %d ms" % (alpha, tap_time)]


def run_science_birds():
    """Starts the Angry Birds simulation software (if it isn't running already)."""
    print("Starting Science Birds...")
    if "Science Birds.exe" not in (p.NAME() for p in psutil.process_iter()):
        subprocess.Popen(["src/envs/ab/Science Birds 0.3.8/Science Birds.exe"],
                         cwd="src/envs/ab/Science Birds 0.3.8/")


class AngryBirds(ParallelEnvironment):
    """A wrapper class for the Science Birds environment."""
    NAME = "angry_birds"
    LEVELS = True
    TIME_RELEVANT = False
    WINS_RELEVANT = True

    def __init__(self, num_par_envs):
        if num_par_envs > 1:
            raise ValueError("ERROR: Yet, only one Angry Birds environment is allowed at the same time. "
                             "You tried to initialize %d parallel environments." % num_par_envs)

        super().__init__(num_par_envs, ACTIONS)

        self.id = None
        self.comm_interface = None
        self.observer = None
        self.framework_process = None
        atexit.register(self.__del__)

        self.validation_levels = []
        self.demo_levels = []
        self.mode = "train"  # level selection mode: training, testing, validation, demo

        self.run_framework()
        run_science_birds()
        self.setup_connections()

        self.set_sim_speed(100)

        print("Initialized Angry Birds successfully!")

    def run_framework(self):
        """Starts the server which communicates between Science Birds and the agent."""
        print("Starting the framework...")
        self.framework_process = subprocess.Popen(['java', '-jar', 'game_playing_interface.jar'],
                                                  stdout=open(os.devnull, 'w'),
                                                  cwd="src/envs/ab/AB Framework 0.3.8/")

    def __del__(self):
        self.framework_process.terminate()
        print("Deleted Angry Birds environment.")

    def setup_connections(self):
        self.id = 2888

        host = "127.0.0.1"
        self.comm_interface = AgentClient(host, "2004", **SERVER_CLIENT_CONFIG)
        self.observer = AgentClient(host, "2006", **SERVER_CLIENT_CONFIG)

        print("Connecting agent to server...")
        try:
            self.comm_interface.connect_to_server()
        except socket.error as e:
            print("Error in client-server communication: " + str(e))

        print("Connecting observer agent to server...")
        try:
            self.observer.connect_to_server()
        except socket.error as e:
            print("Error in client-server communication: " + str(e))

        self.comm_interface.configure(self.id)
        self.observer.configure(self.id)

    def reset(self):
        self.load_next_level()
        self.times[:] = 0
        self.game_overs[:] = False

    def reset_for(self, ids):
        self.load_next_level()
        self.times[:] = 0
        self.game_overs[:] = False

    def set_mode(self, mode):
        """Sets the environment's level selection mode. There are four options:
         - 'train': selects non-validation levels randomly
         - 'test': selects any level randomly
         - 'validate': selects only validation levels
         - 'demo': selects only demo levels"""
        if mode in ["train", "test", "validate", "demo"]:
            self.mode = mode
        else:
            raise ValueError("ERROR: Invalid mode option given. You provided %s but only "
                             "'train', 'test', 'validate', and 'demo' are allowed." % str(mode))

    def load_next_level(self):
        """Loads a level, depending on the level selection mode."""

        if self.mode == "train":
            # Pick a random non-validation level
            non_validation_levels = np.delete(range(1, TOTAL_LEVEL_NUMBER + 1), self.validation_levels)
            next_level = np.random.choice(non_validation_levels)

        elif self.mode == "test":
            # Pick any level randomly
            next_level = np.random.randint(TOTAL_LEVEL_NUMBER) + 1

        elif self.mode == "validate":
            # Pick any validation level randomly
            next_level = np.random.choice(self.validation_levels)

        else:
            # Pick a random demo level
            next_level = np.random.choice(self.demo_levels)

        self.comm_interface.load_level(next_level)

    def load_specified_level(self, level_number=None):
        self.comm_interface.load_level(level_number)

    def step(self, actions):
        _, score, appl_state = self.perform_actions(actions)
        score = np.array([score], dtype='uint')
        game_over = (appl_state == GameState.WON or appl_state == GameState.LOST)
        reward = score2reward(score)
        self.times += 1
        self.game_overs[:] = game_over
        return reward, score, self.game_overs, self.times, self.wins

    def perform_actions(self, action):
        """Performs a shot and observes and returns the consequences."""

        # Convert action index into aim vector and tap time
        alpha, tap_time = action_to_params(action)

        # Perform the shot
        sling_x = 191
        sling_y = 344
        self.comm_interface.shoot(sling_x, sling_y, alpha, 0, 1, tap_time, isPolar=True)

        # Get the environment state (cropped screenshot)
        env_state = self.get_states()

        # Obtain game score
        score = self.comm_interface.get_current_score()

        # Get the application state
        appl_state = self.comm_interface.get_game_state()

        return env_state, score, appl_state

    def get_states(self):
        # Obtain game screenshot
        screenshot, ground_truth = self.comm_interface.get_ground_truth_with_screenshot()

        # Update Vision (to get an up-to-date sling reference point)
        # self.vision.update(screenshot, ground_truth)

        # Crop the image to reduce information overload.
        # The cropped image has then dimension (325, 800, 3).
        crop = screenshot[75:400, 40:]

        # Rescale the image into a (smaller) square
        scaled = cv2.resize(crop, (STATE_PIXEL_RES, STATE_PIXEL_RES))

        # Convert into unsigned byte
        state = np.expand_dims(scaled.astype(np.uint8), axis=0)

        return state, []

    def get_state_shapes(self):
        image_state_shape = [STATE_PIXEL_RES, STATE_PIXEL_RES, 3]
        numerical_state_shape = 0
        return image_state_shape, numerical_state_shape

    def get_number_of_actions(self):
        return len(self.actions)

    def set_sim_speed(self, speed):
        self.comm_interface.set_game_simulation_speed(speed)

    def print_state(self, state):
        fig = plt.imshow(state)
        fig.axes.get_xaxis().set_visible(False)
        fig.axes.get_yaxis().set_visible(False)
        # plt.savefig("plots/state.png", dpi=400)
        plt.show()
        return ""


def score2reward(score):
    """Turns scores into rewards."""
    reward = score / SCORE_NORMALIZATION
    return reward
