from src.utils.stats import compare_statistics
from src.envs import *

compare_statistics(["titus", "negative_a", "prioritized", "smaller_replay"],
                   env_type=Snake,
                   cut_at_episode=50000, cut_at_transition=20000000,
                   cut_at_cycle=1000, cut_at_hour=2)

# cb = ChainBomb(1)
# highscores_ai, highscores_human = cb.get_highscores()
# plot_highscores(highscores_ai, highscores_human)
