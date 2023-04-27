from src.utils.stats import compare_statistics
from src.envs import *

compare_statistics(
    ["add_dense_2b",
     "adjust_lr_v2_2",
     "adjust_lr_v3_5",
     "latent_64",
     "remove_dense"],
    env_type=Tetris,
    # cut_at_episode=300000,
    # cut_at_transition=200000000,
    # cut_at_hour=30
)
