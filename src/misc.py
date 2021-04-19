from src.utils.stats import compare_statistics
from src.envs import *
# "titus", "negative_a", "prioritized", "eps_400K"
compare_statistics(["double_lr", "step_lr_04", "step_lr_v1_2", "step_lr_04_b"],
                   env_type=Snake,
                   cut_at_episode=700000, cut_at_transition=200000000,
                   cut_at_cycle=7000, cut_at_hour=15)

# cb = ChainBomb(1)
# highscores_ai, highscores_human = cb.get_highscores()
# plot_highscores(highscores_ai, highscores_human)
