from src.utils.stats import compare_statistics
from src.envs import *

compare_statistics(["double_lr_2", "smaller_conv"],
                   env=Snake)

# cb = ChainBomb(1)
# highscores_ai, highscores_human = cb.get_highscores()
# plot_highscores(highscores_ai, highscores_human)
