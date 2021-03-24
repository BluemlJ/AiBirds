from src.agents.dqn import *
from src.envs import *

num_parallel_envs = 500
replay_period = 64
sync_period = 256
replay_size = int((num_parallel_envs * replay_period) * 4)
batch_size = 1024
replay_epochs = 1

warmup_cycles = 0
warmup_epochs = warmup_cycles * replay_epochs
warmup_batches = int(warmup_epochs * replay_size / batch_size)

obs_buf_size = 4000  # number of transitions that can fit into the observations buffer per env, = max episode length +1
exp_buf_size = 4000000  # total number of transitions that can fit into the agent's replay memory

# load_and_play("finnson_improved", Snake)

agent = TFDQNAgent(env=Snake,
                   num_parallel_envs=num_parallel_envs,  # look table for fastest value
                   name="residual_v2_double_sync",
                   use_dueling=True,
                   use_double=True,
                   learning_rate=0.0001,  # not larger than 0.0001 (tested)
                   warmup_batches=warmup_batches,
                   latent_dim=256,
                   latent_depth=1,
                   latent_a_dim=128,
                   latent_v_dim=128,
                   obs_buf_size=obs_buf_size,
                   exp_buf_size=exp_buf_size,
                   use_pretrained=False)

# agent.restore("pretrainedguy_2")

agent.practice(num_parallel_steps=1000000,
               replay_period=replay_period,
               replay_size=replay_size,
               batch_size=batch_size,
               replay_epochs=replay_epochs,
               sync_period=sync_period,
               gamma=0.999,
               epsilon=1,
               epsilon_decay_mode="exp",
               epsilon_decay_rate=0.9996,
               epsilon_min=0,
               delta=0,
               delta_anneal=1,
               alpha=0.7,
               verbose=True)

'''print("\nTerminals:")
ids = np.where(agent.memory.get_terminals())[0]
if len(ids):
    agent.print_transitions(ids[:100])

print("\nLast 100:")
length = agent.memory.get_length()
agent.print_transitions(range(length - 100, length))'''
