import gymnasium as gym
import cv2
import numpy as np
from src.envs.env import ParallelEnvironment
import matplotlib.pyplot as plt


RESIZE_DIM = (105, 80)


class Breakout(ParallelEnvironment):
    NAME = "breakout"
    TIME_RELEVANT = True

    def __init__(self, num_par_inst):
        """For env names see: https://gym.openai.com/envs/#atari"""
        actions = ["IDLE", "FIRE", "RIGHT", "LEFT"]
        super(Breakout, self).__init__(num_par_inst, actions)
        self.gym_envs = [gym.make("Breakout-v0") for i in range(num_par_inst)]
        # action_space = self.gym_envs[0].action_space
        state_shape = self.gym_envs[0].observation_space.shape
        self.state_shape = (*RESIZE_DIM, 1)
        self.states = np.zeros(shape=(self.num_par_inst, *self.state_shape))
        self.lives = np.zeros(shape=self.num_par_inst)
        self.no_balls = np.zeros(shape=self.num_par_inst, dtype="bool")

    def reset(self):
        super(Breakout, self).reset()
        for gym_env in self.gym_envs:
            gym_env.reset()
        self.lives[:] = 5
        self.update_states()

    def reset_for(self, ids):
        super(Breakout, self).reset_for(ids)
        for env_id in ids:
            self.gym_envs[env_id].reset()
        self.lives[ids] = 5
        self.update_states(ids)

    def update_states(self, ids=None):
        ids = range(self.num_par_inst) if ids is None else ids
        for env_id in ids:
            # self.states[env_id] = self.gym_envs[env_id].env.state
            self.states[env_id] = self.preprocess(self.gym_envs[env_id].env.ale.getScreenRGB())

    def step(self, actions):  # TODO: implement and test threading
        self.no_balls[actions == 1] = False
        rewards = np.zeros(self.num_par_inst)
        new_lives = np.zeros(self.num_par_inst)

        for i, (action, gym_env) in enumerate(zip(actions, self.gym_envs)):
            observation, reward, done, info = gym_env.step(action)
            new_lives[i] = info["ale.lives"]
            self.states[i] = self.preprocess(observation)
            rewards[i] += reward  # * 0.5
            self.scores[i] += reward
            self.game_overs[i] = done

        ball_lost = new_lives < self.lives
        rewards[ball_lost] -= 1
        rewards[self.no_balls] -= 0.01
        self.no_balls[ball_lost] = True
        self.wins[self.game_overs & ~ ball_lost] = True

        self.lives = new_lives
        self.times += 1

        return rewards, self.scores, self.game_overs, self.times, self.wins

    def preprocess(self, state):
        return np.expand_dims(np.average(cv2.resize(state, RESIZE_DIM[::-1]), axis=2), axis=2) / 255

    def get_states(self):
        return [self.states]

    def get_state_shapes(self):
        return [self.state_shape]

    def get_number_of_actions(self):
        return len(self.actions)

    def render(self):
        assert self.num_par_inst == 1, "Only ParallelEnvironments with a single instance support rendering."
        self.gym_envs[0].render()

    def plot_state(self, state):
        plt.imshow(state)
        plt.show()

    def has_test_levels(self):
        return False

    def generate_pretrain_data(self, num_instances):
        pass

    def set_seed(self, seed):
        for env in self.gym_envs:
            env.seed(seed)
