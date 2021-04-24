from src.utils.stats import compare_statistics
from src.envs import *
# "titus", "negative_a", "prioritized", "eps_400K"
compare_statistics(["eps_0", "new_fruit_spawning", "new_replay_buffer", "debug"],
                   env_type=Snake,
                   cut_at_episode=40000, cut_at_transition=100000000,
                   cut_at_cycle=4000, cut_at_hour=12)

# cb = ChainBomb(1)
# highscores_ai, highscores_human = cb.get_highscores()
# plot_highscores(highscores_ai, highscores_human)
