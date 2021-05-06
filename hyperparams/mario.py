from src.envs import *
import src.agent.comp as comp
from src.utils.params import ParamScheduler
from src.utils.utils import set_seed

seed = 894165
set_seed(seed)

# Environment
env = SuperMario(num_par_inst=1)
env.set_seed(seed)

hyperparams = {
    # General
    "name": "debug",
    "num_parallel_steps": 400000,
    "seed": seed,
    "env": env,

    # Training and synchronization
    "learning_rate": ParamScheduler(init_value=0.001, decay_mode="step",
                                    milestones=[200000],
                                    milestone_values=[0.00025]),
    "replay_period": 32,
    "replay_size_multiplier": 4,
    "replay_batch_size": 256,
    "alpha": 0.5,

    # Learning target returns
    "gamma": 0.99,
    "n_step": 1,

    # Model
    "stem_network": comp.generic.RainbowImproved(1024),
    "q_network": comp.q_network.DoubleQNetwork(128, 128),

    # Policy
    "epsilon": ParamScheduler(init_value=1, decay_mode="lin",
                              milestones=[50000, 1000000, 2000000],
                              milestone_values=[1, 0.1, 0.01]),

    # Miscellaneous
    "memory_size": 400000,
    "stack_size": 4,
}
