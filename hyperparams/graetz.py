from src.envs import *
import src.agent.model as model
from src.utils.params import ParamScheduler
from src.utils.utils import set_seed

seed = 894165
set_seed(seed)

# Environment
env = Pong(num_par_inst=1, frame_skipping=4)
env.set_seed(seed)

hyperparams = {  # max episode length: 18000
    # General
    "name": "graetz",
    "num_parallel_steps": 3000000,
    "seed": seed,
    "env": env,

    # Training and synchronization
    "learning_rate": ParamScheduler(init_value=0.00025),
    "replay_period": 4,
    "replay_size_multiplier": 8,
    "replay_batch_size": 32,
    "alpha": 0,
    "min_hist_len": 50000,
    "max_replay_size": 32,

    # Learning target returns
    "gamma": 0.99,
    "target_sync_period": 10000,

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
}
