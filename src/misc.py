from src.utils.utils import *
from src.envs import *

compare_statistics(["nopretrainedguy", "pretrainedguy_3", "double_latent_depth"], ChainBomb)

# cb = ChainBomb(1)
# highscores_ai, highscores_human = cb.get_highscores()
# plot_highscores(highscores_ai, highscores_human)
