from src.utils.stats import compare_statistics
from src.envs import *

compare_statistics(["double_lr_2", "frank", "frequent_replay", "gamma_992", "john"],
                   env_type=Snake, cut_at_cycle=5000, cut_at_episode=80000)

# cb = ChainBomb(1)
# highscores_ai, highscores_human = cb.get_highscores()
# plot_highscores(highscores_ai, highscores_human)
