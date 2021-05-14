from src.envs import *
import src.agent.model as model
from src.utils.params import ParamScheduler
from src.utils.utils import set_seed

seed = 56498674
set_seed(seed)

# Environment
env = Pong(num_par_inst=50, frame_skipping=4)
env.set_seed(seed)

hyperparams = {
    # General
    "name": "debug",
    "num_parallel_steps": 400000,
    "seed": seed,
    "env": env,

    # Training and synchronization
    "learning_rate": ParamScheduler(init_value=0.00025),
    "replay_period": 16,
    "replay_size_multiplier": 4,
    "replay_batch_size": 256,
    "alpha": 0.5,

    # Learning target returns
    "gamma": 0.99,
    "n_step": 3,
    # "target_sync_period": 200,

    # Model
    "stem_network": model.generic.RainbowImproved(1024),
    "q_network": model.q_network.DoubleQNetwork(128, 128),

    # Policy
    "epsilon": ParamScheduler(init_value=1, decay_mode="lin",
                              milestones=[50000, 1000000, 2000000],
                              milestone_values=[1, 0.1, 0.0]),

    # Miscellaneous
    "memory_size": 1000000,
    "stack_size": 4,
    "max_replay_size": 4000,
}
