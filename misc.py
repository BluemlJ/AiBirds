import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = "2"  # let TF only print errors
from src.utils.stats import compare_statistics
from src.utils.utils import setup_hardware
from src.envs import *
from src.agent.agent import load_and_play

# setup_hardware(use_gpu=False, gpu_memory_limit=4096)
# load_and_play("hendrik", Pong)

# "titus", "negative_a", "prioritized", "eps_400K"
compare_statistics(["hendrik", "lr_epsilon_alpha_2"],
                   env_type=Pong)

# cb = ChainBomb(1)
# highscores_ai, highscores_human = cb.get_highscores()
# plot_highscores(highscores_ai, highscores_human)
