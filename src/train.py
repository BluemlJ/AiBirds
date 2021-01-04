from src.agents.dqn import *
from src.envs import *
import time

num_parallel_envs = 500
replay_period = 8
replay_size = int((num_parallel_envs * replay_period) * 4)

obs_buf_size = 5  # number of transitions that can fit into the observations buffer per env
exp_buf_size = 1000000  # total number of transitions that can fit into the agent's replay memory

# load_and_play("finnson_improved", Snake)

agent = TFDQNAgent(env=ChainBomb,
                   num_parallel_envs=num_parallel_envs,  # look table for fastest value
                   name="double_latent_depth",
                   use_dueling=True,
                   use_double=True,
                   learning_rate=0.0001,  # not larger than 0.0001 (tested)
                   latent_dim=256,
                   latent_a_dim=128,
                   latent_v_dim=128,
                   latent_depth=2,
                   obs_buf_size=obs_buf_size,
                   exp_buf_size=exp_buf_size,
                   use_pretrained=False)

# agent.restore("pretrainedguy_2")

agent.practice(num_parallel_steps=60000,
               replay_period=replay_period,
               replay_size=replay_size,
               batch_size=4096,
               replay_epochs=1,
               sync_period=16,
               gamma=0.99,
               epsilon=1,
               epsilon_decay_mode="exp",
               epsilon_decay_rate=0.9995,
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
