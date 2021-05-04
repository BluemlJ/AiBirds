from src.envs import *
import src.agent.comp as comp
from src.utils.utils import setup_hardware, set_seed
import tensorflow as tf
from src.utils.params import ParamScheduler

seed = 894165
set_seed(seed)

# Environment
num_par_envs = 50
env = Pong(num_par_inst=num_par_envs)
env.set_seed(seed)

hyperparams = {
    # General
    "name": "rainbow",
    "num_parallel_steps": 100000,
    "seed": seed,
    "env": env,

    # Training and synchronization
    "optimizer": tf.optimizers.Adam(epsilon=1.5e-4),
    "learning_rate": ParamScheduler(init_value=0.0000625),
    "min_hist_len": 80000,
    "replay_period": 4,
    "replay_size_multiplier": 8,
    "replay_batch_size": 128,
    "alpha": 0.5,
    "target_sync_period": 10000 // num_par_envs,

    # Learning target returns
    "gamma": 0.99,
    "n_step": 3,

    # Model
    "stem_network": comp.generic.Rainbow(),
    "q_network": comp.q_network.DoubleQNetwork(512, 512),

    # Policy
    "epsilon": ParamScheduler(init_value=1, decay_mode="lin",
                              milestones=[50000, 1000000, 2000000],
                              milestone_values=[1, 0.1, 0.01]),

    # Miscellaneous
    "memory_size": 200000,
    "stack_size": 4,
}
