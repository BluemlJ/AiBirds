from src.envs.tetris import Tetris
from src.envs.snake import Snake
from src.envs.angry_birds import AngryBirds
from src.envs.chain_bomb import ChainBomb
from src.envs.chain_bomb import CBCreator

cb = ChainBomb(1)
cb.set_mode(cb.TEST_MODE)
cb.run_for_human()

# creator = CBCreator()
