from src.utils.stats import compare_statistics
from src.envs import *
# "titus", "negative_a", "prioritized", "eps_400K"
compare_statistics(["new_fruit_spawning", "20_step_a", "1_step", "5_step", "2_step",
                    "5_step_epsilon"],
                   env_type=Snake,
                   cut_at_episode=40000, cut_at_transition=25000000,
                   cut_at_cycle=1000, cut_at_hour=4)

# cb = ChainBomb(1)
# highscores_ai, highscores_human = cb.get_highscores()
# plot_highscores(highscores_ai, highscores_human)
