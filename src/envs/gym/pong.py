import gym
import cv2
import numpy as np
from src.envs.env import ParallelEnvironment
from src.utils.utils import plot_grayscale
from src.utils.render import spread_windows


RESIZE_DIM = (84, 84)
MAX_EPISODE_LEN = 1000


class Pong(ParallelEnvironment):
    NAME = "pong"
    TIME_RELEVANT = False
    WINS_RELEVANT = True

    def __init__(self, num_par_inst):
        """For env names see: https://gym.openai.com/envs/#atari"""
        actions = ["IDLE", "UP", "DOWN"]
        super(Pong, self).__init__(num_par_inst, actions)
        self.gym_envs = [gym.make("Pong-v0") for i in range(num_par_inst)]
        self.state_shape = (*RESIZE_DIM, 1)
        self.states = np.zeros(shape=(self.num_par_inst, *self.state_shape))
        self.update_states()
        self.windows_initialized = False

    def reset(self):
        super(Pong, self).reset()
        for gym_env in self.gym_envs:
            gym_env.reset()
        self.update_states()

    def reset_for(self, ids):
        super(Pong, self).reset_for(ids)
        for env_id in ids:
            self.gym_envs[env_id].reset()
        self.update_states(ids)

    def update_states(self, ids=None):
        ids = range(self.num_par_inst) if ids is None else ids
        for env_id in ids:
            self.states[env_id] = self.fetch_state(env_id)

    def fetch_state(self, env_id):
        return self.preprocess(self.gym_envs[env_id].env.ale.getScreenGrayscale())

    def preprocess(self, state):
        state = cv2.resize(state, RESIZE_DIM, interpolation=cv2.INTER_LINEAR).astype("float32") / 255
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

    def get_number_of_actions(self):
        return len(self.actions)

    def render(self):
        if self.num_par_inst == 1:
            self.gym_envs[0].render()
        else:
            self.display_all()

    def display_all(self):
        if not self.windows_initialized:
            self.init_windows()
        for env_id in range(self.num_par_inst):
            cv2.imshow('Env %d' % env_id, self.states[env_id])
            cv2.waitKey(1)

    def plot_all_states(self):
        for state in self.states:
            plot_grayscale(state)

    def plot_state(self, idx):
        x_label = "Time = %d, Score = %d, Game Over = %s" % (self.times[idx], self.scores[idx], self.game_overs[idx])
        plot_grayscale(self.states[idx], title="State of env %d" % idx, x_label=x_label)

    def init_windows(self):
        window_placements = spread_windows(self.num_par_inst)
        for env_id in range(self.num_par_inst):
            win_name = 'Env %d' % env_id
            cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)
            x, y, w, h = window_placements[env_id]
            cv2.moveWindow(win_name, x, y)
            cv2.resizeWindow(win_name, w, h)
        self.windows_initialized = True

    def has_test_levels(self):
        return False

    def generate_pretrain_data(self, num_instances):
        pass

    def set_seed(self, seed):
        np.random.seed(seed)
        env_seeds = np.random.randint(0, high=1e9, size=self.num_par_inst)
        for env_seed, env in zip(env_seeds, self.gym_envs):
            env.seed(int(env_seed))
