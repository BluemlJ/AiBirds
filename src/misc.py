from src.utils.stats import compare_statistics
from src.envs import *
# "titus", "negative_a", "prioritized", "eps_400K"
compare_statistics(["double_lr", "bool_b", "eps_0_2", "new_fruit_spawning"],
                   env_type=Snake,
                   cut_at_episode=40000, cut_at_transition=30000000,
                   cut_at_cycle=1000, cut_at_hour=2)

# cb = ChainBomb(1)
# highscores_ai, highscores_human = cb.get_highscores()
# plot_highscores(highscores_ai, highscores_human)
