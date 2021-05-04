from src.agent.agent import *
from src.utils.utils import setup_hardware

# Import hyperparameters dictionary here
from hyperparams.pong import hyperparams

setup_hardware(use_gpu=True, gpu_memory_limit=4096)

agent = Agent(**hyperparams)

# agent.restore("lr_epsilon_alpha")

agent.practice(**hyperparams)
