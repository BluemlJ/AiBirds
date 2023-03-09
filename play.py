from src.envs import *
from src.utils.utils import setup_hardware
from src.agent.agent import load_and_play

setup_hardware(use_gpu=False, gpu_memory_limit=4096)
load_and_play("half_latent_2", Tetris, num_par_envs=1, verbose=True)
