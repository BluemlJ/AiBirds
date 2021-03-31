from src.utils.utils import *
from src.envs import *

compare_statistics(["finnson_3", "gamma_9999", "half_latent"],
                   cut_at_episode=1000000, env=Tetris)

# cb = ChainBomb(1)
# highscores_ai, highscores_human = cb.get_highscores()
# plot_highscores(highscores_ai, highscores_human)
