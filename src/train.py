from src.agents.dqn import *
from src.envs.tetris import Tetris
from src.envs.snake import Snake
from src.envs.angry_birds import AngryBirds
from src.envs.chain_bomb import ChainBomb
import time

num_parallel_envs = 1
replay_period = 64
replay_size = int((num_parallel_envs * replay_period) * 4)

obs_buf_size = 10  # number of transitions that can fit into the observations buffer per env
exp_buf_size = 10000  # total number of transitions that can fit into the agent's replay memory

# load_and_play("finnson_improved", Snake)

agent = TFDQNAgent(env=ChainBomb,
                   num_parallel_envs=num_parallel_envs,  # look table for fastest value
                   name="no_lr_decrease_3",
                   use_dueling=True,
                   use_double=True,
                   learning_rate=0.0001,  # not larger than 0.0001 (tested)
                   latent_dim=256,
                   latent_a_dim=128,
                   latent_v_dim=128,
                   obs_buf_size=obs_buf_size,
                   exp_buf_size=exp_buf_size)

agent.restore("no_lr_decrease_2")

agent.practice(num_parallel_steps=1000000,
               replay_period=replay_period,
               replay_size=replay_size,
               batch_size=4096,
               replay_epochs=1,
               sync_period=128,
               gamma=0.99,
               epsilon=0,
               epsilon_decay_mode="exp",
               epsilon_decay_rate=0.99995,
               epsilon_min=0,
               delta=0,
               delta_anneal=1,
               alpha=0.7,
               verbose=False)

"""# print("Positive reward:")
ids = np.where(agent.memory.get_rewards() > 0)[0]
if len(ids):
    agent.print_transitions(ids)

# print("Terminals:")
ids = np.where(agent.memory.get_terminals())[0]
if len(ids):
    agent.print_transitions(ids)

print("Last 100:")
length = agent.memory.get_length()
agent.print_transitions(range(length - 100, length))"""
