from src.envs import *
import src.agent.model as model
from src.utils.utils import set_seed
from src.utils.params import ParamScheduler
import numpy as np

seed = np.random.randint(1e9)
set_seed(seed)

# Environment
env = Snake(num_par_inst=500)
env.set_seed(seed)

hyperparams = {
    # General
    "name": "no_split_no_ReLu",
    "num_parallel_steps": 1000000,
    "seed": seed,
    "env": env,

    # Training and synchronization
    "learning_rate": ParamScheduler(init_value=0.0005, decay_mode="step",
                                    milestones=[5000000, 50000000],
                                    milestone_values=[0.0002, 0.00008]),
    "replay_period": 64,
    "replay_size_multiplier": 4,
    "replay_epochs": 1,
    "replay_batch_size": 1024,
    "alpha": 0.7,

    # Learning target returns
    "gamma": 0.999,

    # Model
    "stem_network": model.generic.StemNetwork2D1D(latent_dim=128),
    "q_network": model.q_network.DoubleQNetwork(),

    # Policy
    "epsilon": ParamScheduler(init_value=0),

    # Miscellaneous
    "memory_size": 4000000,
    "stack_size": 4,
}
