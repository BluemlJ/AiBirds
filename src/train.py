from src.agents.agent import *
from src.envs import *
import src.agents.comp as comp
from src.utils.utils import setup_hardware, set_seed

# Meta
setup_hardware(use_gpu=True, gpu_memory_limit=4096)
seed = 486217935
set_seed(seed)

# General parameters
env = Snake(num_par_inst=500)

# Training and synchronization
replay_period = 64
replay_size_multiplier = 2  # multiplier of 4 means that, per replay on average, 25 % of data is unseen.
replay_epochs = 1
replay_batch_size = 1024
target_sync_period = 128
actor_sync_period = replay_period
learning_rate = DecayParam(init_value=0.0004, decay_mode="exp", half_life_period=4000000, warmup_transitions=0)
delta = DecayParam(init_value=0)

# Model with recurrence
sequence_len = 20
eta = 0.9

# Stem model
latent_dim = 128
latent_depth = 1  # number of latent (dense) layers (if supported)
stem_model = comp.generic.ConvStemNetwork(latent_dim=latent_dim)  # , lstm_dim=latent_dim, sequence_len=sequence_len)

# Q-network
latent_v_dim = 64  # dimension of value part of q-network
latent_a_dim = 64  # dimension of advantage part of q-network
q_network = comp.q_network.DoubleQNetwork(latent_v_dim, latent_a_dim)

# Policy
epsilon = DecayParam(init_value=1, decay_mode="exp", half_life_period=700000)

# Miscellaneous
obs_buf_size = 2000  # number of transitions that can fit into the observations buffer per env, = max episode length +1
exp_buf_size = 4000000  # total number of transitions that can fit into the agent's replay memory

# load_and_play("frank", Snake, checkpoint_no=42842)

agent = Agent(env=env,
              stem_network=stem_model,
              q_network=q_network,
              name="smaller_replay_no_prio",
              replay_batch_size=replay_batch_size,
              sequence_shift=10,
              eta=eta,
              use_double=False,
              obs_buf_size=obs_buf_size,
              mem_size=exp_buf_size,
              use_pretrained=False,
              seed=seed)
# agent.restore("lstm")

agent.practice(num_parallel_steps=1000000,
               replay_period=replay_period,
               replay_size_multiplier=replay_size_multiplier,
               replay_epochs=replay_epochs,
               learning_rate=learning_rate,
               target_sync_period=target_sync_period,
               actor_sync_period=actor_sync_period,
               gamma=0.999,
               epsilon=epsilon,
               delta=delta,
               alpha=0,
               verbose=False)

'''print("\nTerminals:")
ids = np.where(agent.memory.get_terminals())[0]
if len(ids):
    agent.print_transitions(ids[:100])

print("\nLast 100:")
length = agent.memory.get_length()
agent.print_transitions(range(length - 100, length))'''
