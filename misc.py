import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = "2"  # let TF only print errors
from src.utils.stats import compare_statistics
from src.envs import *
from src.utils.utils import setup_hardware
from src.agent.agent import load_and_play

setup_hardware(use_gpu=True, gpu_memory_limit=4096)
load_and_play("deterministic_frame_skipping_4", Pong, num_par_envs=8, epsilon=0.01)

# compare_statistics(["larger_lr", "gamma", "constant_lr"],
#                    env_type=Pong)

# cb = ChainBomb(1)
# highscores_ai, highscores_human = cb.get_highscores()
# plot_highscores(highscores_ai, highscores_human)
