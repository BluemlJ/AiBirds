from src.envs import *
from src.utils.utils import setup_hardware
from src.agent.agent import play

setup_hardware(use_gpu=True, gpu_memory_limit=4096)
play("adjust_lr_v3_5",
     Tetris,
     num_par_envs=4,
     verbose=True,
     render_environment=True)
