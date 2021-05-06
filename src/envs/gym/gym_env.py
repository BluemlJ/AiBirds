import numpy as np
from abc import ABCMeta
from src.envs.env import ParallelEnvironment
from src.utils.render import ParallelScreen


class ParGymEnv(ParallelEnvironment, metaclass=ABCMeta):
    def __init__(self, num_par_inst, gym_envs, actions):
        super(ParGymEnv, self).__init__(num_par_inst, actions)
        self.gym_envs = gym_envs

    def reset(self, env_ids=None, **kwargs):
        super(ParGymEnv, self).reset(env_ids)
        if env_ids is None:
            env_ids = np.arange(self.num_par_inst)
        for env_id in env_ids:
            self.gym_envs[env_id].reset()

    def set_seed(self, seed):
        np.random.seed(seed)
        env_seeds = np.random.randint(0, high=1e9, size=self.num_par_inst)
        for env_seed, env in zip(env_seeds, self.gym_envs):
            env.seed(int(env_seed))

    def has_test_levels(self):
        return False

    def generate_pretrain_data(self, num_instances):
        pass


class ParScreenGymEnv(ParGymEnv, metaclass=ABCMeta):
    def __init__(self, num_par_inst, screen_shape, **kwargs):
        super(ParScreenGymEnv, self).__init__(num_par_inst, **kwargs)
        self.parallel_screen = ParallelScreen(num_par_inst, screen_shape)

    def render(self):
        self.parallel_screen.render()
