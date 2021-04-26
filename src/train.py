from src.agents.agent import *
from src.envs import *
import src.agents.comp as comp
from src.utils.utils import setup_hardware, set_seed

# Meta
setup_hardware(use_gpu=True, gpu_memory_limit=4096)
seed = 735249652
set_seed(seed)

# Environment
# env = Snake(num_par_inst=500)
env = Breakout(num_par_inst=20)

# Training and synchronization
replay_period = 64
replay_size_multiplier = 4  # multiplier of 4 means that, per replay on average, 25 % of data is unseen.
replay_epochs = 1
replay_batch_size = 256
target_sync_period = 128
actor_sync_period = replay_period
# learning_rate = ParamScheduler(init_value=0.00008, warmup_transitions=400000)
learning_rate = ParamScheduler(init_value=0.0005, decay_mode="step", milestones=[30000, 100000],
                               milestone_factor=0.4)

# Learning target
gamma = 0.99
n_step = 1
use_mc_return = False

# Model with recurrence
sequence_len = 20
sequence_shift = 10
eta = 0.9

# Stem model
latent_dim = 128
latent_depth = 1  # number of latent (dense) layers (if supported)
stem_model = comp.generic.StemNetwork2D(latent_dim=latent_dim)  # , lstm_dim=latent_dim, sequence_len=sequence_len)

# Q-network
latent_v_dim = 64  # dimension of value part of q-network
latent_a_dim = 64  # dimension of advantage part of q-network
q_network = comp.q_network.DoubleQNetwork(latent_v_dim, latent_a_dim)

# Policy
epsilon = ParamScheduler(init_value=1, decay_mode="exp", half_life_period=30000)

# Miscellaneous
exp_buf_size = 20000  # total number of transitions that can fit into the agent's replay memory

agent = Agent(env=env,
              stem_network=stem_model,
              q_network=q_network,
              name="grayscale",
              replay_batch_size=replay_batch_size,
              use_double=False,
              use_pretrained=False,
              seed=seed)
# agent.restore("new_fruit_spawning")
# agent.just_play(verbose=True, epsilon=1)

agent.practice(num_parallel_steps=1000000,
               replay_period=replay_period,
               replay_size_multiplier=replay_size_multiplier,
               replay_epochs=replay_epochs,
               learning_rate=learning_rate,
               target_sync_period=target_sync_period,
               actor_sync_period=actor_sync_period,
               gamma=gamma,
               epsilon=epsilon,
               use_mc_return=use_mc_return,
               alpha=0.7,
               memory_size=exp_buf_size,
               n_step=n_step,
               sequence_shift=sequence_shift,
               eta=eta,
               verbose=False)

'''print("\nTerminals:")
ids = np.where(agent.memory.get_terminals())[0]
if len(ids):
    agent.print_transitions(ids[:100])

print("\nLast 100:")
length = agent.memory.get_length()
agent.print_transitions(range(length - 100, length))'''
