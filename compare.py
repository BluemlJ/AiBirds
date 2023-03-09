from src.utils.stats import compare_statistics
from src.envs import *

compare_statistics(["finnson_3", "half_latent_2", "no_relu_b", "no_dense_b", "larger_lr",
                    "even_larger_lr"],
                   env_type=Tetris,
                   cut_at_transition=300000000)
