import gym
import cv2
import numpy as np
from src.envs.gym.gym_env import ParScreenGymEnv


RESIZE_DIM = (84, 84)
MAX_EPISODE_LEN = 2000


class Pong(ParScreenGymEnv):
    NAME = "pong"
    TIME_RELEVANT = False
    WINS_RELEVANT = True

    def __init__(self, num_par_inst):
        """For env names see: https://gym.openai.com/envs/#atari"""
        actions = ["IDLE", "UP", "DOWN"]
        envs = [gym.make("Pong-v0") for i in range(num_par_inst)]
        screen_shape = envs[0].observation_space.shape
        super(Pong, self).__init__(num_par_inst=num_par_inst, gym_envs=envs, actions=actions,
                                   screen_shape=screen_shape)

        self.state_shape = (*RESIZE_DIM, 1)
        self.states = np.zeros(shape=(self.num_par_inst, *self.state_shape))
        self.update_states()

    def update_states(self, ids=None):
        ids = range(self.num_par_inst) if ids is None else ids
        for env_id in ids:
            self.states[env_id] = self.fetch_state(env_id)

    def fetch_state(self, env_id):
        screen = self.gym_envs[env_id].env.ale.getScreenRGB()
        self.parallel_screen.update_screens(screen, env_id)
        return self.preprocess(screen)

    def preprocess(self, screen):
        gray_screen = cv2.cvtColor(screen, cv2.COLOR_BGR2GRAY)
        state = cv2.resize(gray_screen, RESIZE_DIM, interpolation=cv2.INTER_LINEAR).astype("float32") / 255
        return np.expand_dims(state, axis=-1)

    def step(self, actions):  # TODO: implement and test threading
        rewards = np.zeros(self.num_par_inst)
        actions = actions + 1  # convert into original action space

        for i, (action, gym_env) in enumerate(zip(actions, self.gym_envs)):
            observation, reward, done, info = gym_env.step(action)
            self.states[i] = self.fetch_state(i)
            rewards[i] += reward  # * 0.5
            self.scores[i] += reward
            self.game_overs[i] = done

        self.times += 1
        self.game_overs = self.times >= MAX_EPISODE_LEN
        goal_against = rewards < 0
        terminals = self.game_overs | goal_against
        self.wins[self.game_overs] = self.scores[self.game_overs] > 0

        # self.render()

        return rewards, self.scores, terminals, self.times, self.wins, self.game_overs

    def get_states(self):
        return [self.states]

    def get_state_shapes(self):
        return [self.state_shape]
