from nes_py.wrappers import JoypadSpace
import gym_super_mario_bros
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
import cv2
import numpy as np
from src.envs.gym.gym_env import ParScreenGymEnv

RESIZE_DIM = (84, 84)
ACTION_BITMAP = [0b10000000,
                 0b01000000,
                 0b00100000,
                 0b00010000,
                 0b00000010,
                 0b00000001,
                 0b00000000]

# Original action bitmaps:
# 'right':  0b10000000,
# 'left':   0b01000000,
# 'down':   0b00100000,
# 'up':     0b00010000,
# 'start':  0b00001000,
# 'select': 0b00000100,
# 'B':      0b00000010,
# 'A':      0b00000001,
# 'NOOP':   0b00000000,


class SuperMario(ParScreenGymEnv):
    NAME = "mario"
    TIME_RELEVANT = True
    WINS_RELEVANT = True

    def __init__(self, num_par_inst):
        """For env names see: https://gym.openai.com/envs/#atari"""
        # Construct environments
        actions = ["NOOP", "RIGHT", "RIGHT_A", "RIGHT_B", "RIGHT_A_B", "A", "DOWN"]
        envs = []
        for i in range(num_par_inst):
            env = gym_super_mario_bros.make('SuperMarioBros-v0')
            envs += [JoypadSpace(env, SIMPLE_MOVEMENT)]
        screen_shape = envs[0].screen.shape
        super(SuperMario, self).__init__(num_par_inst=num_par_inst, gym_envs=envs, actions=actions,
                                         screen_shape=screen_shape)

        self.state_shape = (*RESIZE_DIM, 1)
        self.states = np.zeros(shape=(self.num_par_inst, *self.state_shape))
        self.update_states()

    def update_states(self, ids=None):
        ids = range(self.num_par_inst) if ids is None else ids
        for env_id in ids:
            self.states[env_id] = self.fetch_state(env_id)

    def fetch_state(self, env_id):
        screen = self.gym_envs[env_id].screen
        self.parallel_screen.update_screens(screen, env_id)
        return self.preprocess(screen)

    def preprocess(self, screen):
        gray_screen = cv2.cvtColor(screen, cv2.COLOR_BGR2GRAY)
        state = cv2.resize(gray_screen, RESIZE_DIM, interpolation=cv2.INTER_LINEAR).astype("float32") / 255
        return np.expand_dims(state, axis=-1)

    def step(self, actions):
        # action_bitmaps = action_ids_to_bitmap(actions)
        rewards = np.zeros(self.num_par_inst)

        for i, (action, gym_env) in enumerate(zip(actions, self.gym_envs)):
            observation, reward, done, info = gym_env.step(action)
            self.states[i] = self.fetch_state(i)
            rewards[i] += reward
            self.scores[i] += reward
            self.game_overs[i] = done
            self.wins[i] = info["flag_get"]

        self.times += 1
        terminals = self.game_overs

        return rewards, self.scores, terminals, self.times, self.wins, self.game_overs

    def get_states(self):
        return [self.states]

    def get_state_shapes(self):
        return [self.state_shape]


# def action_ids_to_bitmap(action_ids):
#     return np.array(ACTION_BITMAP)[action_ids].astype("uint8")
