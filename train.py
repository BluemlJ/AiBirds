import tracemalloc

from src.agent.agent import Agent, continue_practice
from src.utils.utils import setup_hardware, split_params
from src.utils.params import ParamScheduler
from src.envs import Tetris

# Import hyperparameters dictionary here
from hyperparams.tetris import hyperparams

# tracemalloc.start()  # Memory allocation tracking

setup_hardware(use_gpu=True, gpu_memory_limit=4096)

continue_practice("remove_dense", Tetris)

# Split hyperparams into hyperparams for agent initialization and for agent practice
# hparams_agent, hparams_practice = split_params(hyperparams, Agent.__init__)
#
# agent = Agent(**hparams_agent)
# agent.practice(**hparams_practice)
