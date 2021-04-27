import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = "2"  # let TF only print errors
from src.utils.stats import compare_statistics
from src.envs import *
from src.agents.agent import load_and_play

# load_and_play("test", Pong, epsilon=1)

# "titus", "negative_a", "prioritized", "eps_400K"
compare_statistics(["new_replay_buffer", "increasingly_bad_death", "4_stacked", "1_stacked"],
                   env_type=Snake,
                   cut_at_episode=40000, cut_at_transition=10000000,
                   cut_at_cycle=300, cut_at_hour=2)

# cb = ChainBomb(1)
# highscores_ai, highscores_human = cb.get_highscores()
# plot_highscores(highscores_ai, highscores_human)
