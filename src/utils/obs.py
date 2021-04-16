import numpy as np


class Observations:
    """A finite and efficient ring buffer temporally holding observed transitions."""

    def __init__(self, buffer_size, num_envs, state_shape, hidden_state_shape=None):
        self.size = buffer_size

        self.sequential = hidden_state_shape is not None

        state_2d_shape, state_1d_shape = state_shape
        self.states_2d = np.zeros(np.append([buffer_size, num_envs], state_2d_shape), dtype='float32')
        self.states_1d = np.zeros(np.append([buffer_size, num_envs], state_1d_shape), dtype='float32')
        if self.sequential:
            self.hidden_states = np.zeros(np.append([buffer_size, num_envs], hidden_state_shape), dtype='float32')
        self.actions = np.zeros((buffer_size, num_envs), dtype='int')
        self.score_gains = np.zeros((buffer_size, num_envs), dtype='int')
        self.rewards = np.zeros((buffer_size, num_envs), dtype='float32')
        self.times = np.zeros((buffer_size, num_envs), dtype='uint')

        self.buff_ptr = 0  # the buffer pointer, pointing at the next transition's position
        self.ep_beg_ptrs = np.zeros(num_envs, dtype='int')  # pointing at each episode's first transition

        self.curr_scores = np.zeros(num_envs, dtype='int')
        self.curr_returns = np.zeros(num_envs, dtype='float')

    def save_observations(self, states, hidden_states, actions, scores, rewards, times):
        # preprocess data
        states_2d, states_1d = states
        score_gains = scores - self.curr_scores

        # save data
        self.states_2d[self.buff_ptr] = states_2d
        self.states_1d[self.buff_ptr] = states_1d
        if self.sequential:
            self.hidden_states[self.buff_ptr] = hidden_states
        self.actions[self.buff_ptr] = actions
        self.score_gains[self.buff_ptr] = score_gains
        self.rewards[self.buff_ptr] = rewards
        self.times[self.buff_ptr] = times

        self.curr_scores[:] = scores
        self.curr_returns += rewards

        # handle pointers
        self.increment()
        self.handle_full_episodes()

    def increment(self):
        self.buff_ptr = (self.buff_ptr + 1) % self.size

    def handle_full_episodes(self):
        max_len_episodes = self.ep_beg_ptrs == self.buff_ptr
        self.ep_beg_ptrs[max_len_episodes] += 1
        self.ep_beg_ptrs[max_len_episodes] %= self.size

    def get_observations(self, idx):
        ep_beg_ptr = self.ep_beg_ptrs[idx]
        hidden_states = None

        if ep_beg_ptr <= self.buff_ptr:
            image_states = self.states_2d[ep_beg_ptr:self.buff_ptr, idx]
            numerical_states = self.states_1d[ep_beg_ptr:self.buff_ptr, idx]
            if self.sequential:
                hidden_states = self.hidden_states[ep_beg_ptr:self.buff_ptr, idx]
            actions = self.actions[ep_beg_ptr:self.buff_ptr, idx]
            score_gains = self.score_gains[ep_beg_ptr:self.buff_ptr, idx]
            rewards = self.rewards[ep_beg_ptr:self.buff_ptr, idx]
            times = self.times[ep_beg_ptr:self.buff_ptr, idx]
        else:
            # Create fancy index for episode entries
            trans_ids = (list(range(ep_beg_ptr, self.size)) + list(range(self.buff_ptr)), idx)
            image_states = self.states_2d[trans_ids]
            numerical_states = self.states_1d[trans_ids]
            if self.sequential:
                hidden_states = self.hidden_states[trans_ids]
            actions = self.actions[trans_ids]
            score_gains = self.score_gains[trans_ids]
            rewards = self.rewards[trans_ids]
            times = self.times[trans_ids]

        states = [image_states, numerical_states]

        return states, hidden_states, actions, score_gains, rewards, times

    def get_performance(self, idx):
        obs_score = self.curr_scores[idx]
        obs_return = self.curr_returns[idx]

        return obs_score, obs_return

    def begin_new_episode_for(self, ids):
        self.ep_beg_ptrs[ids] = self.buff_ptr
        self.curr_scores[ids] = 0
        self.curr_returns[ids] = 0
