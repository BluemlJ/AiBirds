from src.envs import *
import src.agent.model as model
from src.utils.params import ParamScheduler
import tensorflow as tf
import numpy as np
from src.utils.utils import set_seed

seed = np.random.randint(1e9)
set_seed(seed)

# Environment
env = Pong(num_par_inst=50, frame_skipping=4)
env.set_seed(seed)

hyperparams = {
    # General
    "name": "1_variance_scaling_relu",
    "num_parallel_steps": 400000,
    "seed": seed,
    "env": env,

    # Training and synchronization
    # "optimizer": tf.optimizers.RMSprop(),
    "learning_rate": ParamScheduler(init_value=0.00025),
    "replay_period": 32,
    "replay_size_multiplier": 4,
    "replay_batch_size": 256,
    "alpha": 0.5,

    # Learning target returns
    "gamma": 0.98,
    "n_step": 1,
    # "target_sync_period": 200,

    # Model
    "stem_network": model.generic.RainbowImproved(1024),
    "q_network": model.q_network.DoubleQNetwork(),

    # Policy
    "epsilon": ParamScheduler(init_value=1, decay_mode="lin",
                              milestones=[50000, 1000000, 2000000],
                              milestone_values=[1, 0.1, 0.01]),

    # Miscellaneous
    "memory_size": 1000000,
    "stack_size": 4,
    "max_replay_size": 7000,
}
