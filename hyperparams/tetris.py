from src.envs import *
import src.agent.model as model
from src.utils.utils import set_seed
from src.utils.params import ParamScheduler
import numpy as np

seed = np.random.randint(1e9)
set_seed(seed)

# Environment
env = Tetris(num_par_inst=500)
env.set_seed(seed)

hyperparams = {
    # General
    "name": "even_larger_lr_2",
    "num_parallel_steps": 1000000,
    "seed": seed,
    "env": env,

    # Training and synchronization
    "learning_rate": ParamScheduler(init_value=0.0005, decay_mode="step",
                                    milestones=[100000000],
                                    milestone_values=[0.00025]),
    "replay_period": 64,
    "replay_size_multiplier": 4,
    "replay_epochs": 1,
    "replay_batch_size": 1024,
    "alpha": 0.7,

    # Learning target returns
    "gamma": 0.995,

    # Model
    "stem_network": model.generic.StemNetwork2DSmallNoDense(latent_dim=128),
    "q_network": model.q_network.DoubleQNetwork(),

    # Policy
    "epsilon": ParamScheduler(init_value=1, decay_mode="exp", half_life_period=500000),

    # Miscellaneous
    "memory_size": 4000000,
    "stack_size": 4,
}
