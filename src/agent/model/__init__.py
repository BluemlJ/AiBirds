from src.agent.model.q_network import QNetwork, DoubleQNetwork, VanillaQNetwork
from src.agent.model.stem import StemNetwork
from src.agent.model.generic import StemNetwork2D1D, StemNetwork2DLarge, StemNetwork2D, ConvLSTM,\
    Rainbow, RainbowImproved, StemNetwork2DSmallNoDense
from src.agent.model.noisy import NoisyDense
# from src.agent.model.angry_birds import *
# from src.agent.model.chain_bomb import *
from src.agent.model.snake import *
# from src.agent.model.tetris import *


def get_class_from_name(name: str):
    return eval(name)
