from src.utils.autoenc import Autoencoder
from src.envs import *

ae = Autoencoder(env=ChainBomb)
ae.pretrain_model(train_size=2000000, validation_size=10000, batch_size=4096, epochs=30)
