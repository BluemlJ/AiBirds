from src.utils.autoenc import *
from src.envs import *

pretrain_model(env=ChainBomb, train_size=1000000, validation_size=10000)
