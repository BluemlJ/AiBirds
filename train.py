from src.agent.agent import *
from src.utils.utils import setup_hardware, split_params

# Import hyperparameters dictionary here
from hyperparams.tetris import hyperparams

setup_hardware(use_gpu=True, gpu_memory_limit=4096)

# Split hyperparams into hyperparams for agent initialization and for agent practice
hparams_agent, hparams_practice = split_params(hyperparams, Agent.__init__)

agent = Agent(**hparams_agent)

# agent.restore("deterministic_frame_skipping_4_2")

agent.practice(**hparams_practice)
