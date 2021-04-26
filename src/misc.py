from src.utils.stats import compare_statistics
from src.envs import *
from src.agents.agent import *

load_and_play("ball_loss_penalty", Breakout)

# "titus", "negative_a", "prioritized", "eps_400K"
compare_statistics(["ball_loss_penalty"],
                   env_type=Breakout)
                   # cut_at_episode=80000, cut_at_transition=150000000,
                   # cut_at_cycle=3000, cut_at_hour=10)

# cb = ChainBomb(1)
# highscores_ai, highscores_human = cb.get_highscores()
# plot_highscores(highscores_ai, highscores_human)
