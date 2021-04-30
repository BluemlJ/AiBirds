from src.agents.agent import *
from src.envs import *
import src.agents.comp as comp
from src.utils.utils import setup_hardware, set_seed

# Meta
setup_hardware(use_gpu=True, gpu_memory_limit=4096)
seed = 735249652
set_seed(seed)  # TODO: set for envs

# Environment
num_par_envs = 50
env = Pong(num_par_inst=num_par_envs)

# Training and synchronization
optimizer = tf.optimizers.Adam(epsilon=1.5e-4)
learning_rate = ParamScheduler(init_value=0.0000625)  # , decay_mode="step", milestones=[50000, 500000],
# milestone_factor=0.4)
min_hist_len = 80000
replay_period = 4
replay_size_multiplier = 8  # multiplier of 4 means that, per replay on average, 25 % of data is unseen.
max_replay_size = 500
replay_epochs = 1
replay_batch_size = 128
use_double = True
target_sync_period = 32000 // num_par_envs
actor_sync_period = replay_period

# Learning target returns
gamma = 0.99
n_step = 3
use_mc_return = False

# Model with recurrence
sequence_len = 20
sequence_shift = 10
eta = 0.9

# Stem model
latent_dim = 128
latent_depth = 1  # number of latent (dense) layers (if supported)
stem_model = comp.generic.Rainbow(latent_dim=latent_dim)  # , lstm_dim=latent_dim, sequence_len=sequence_len)

# Q-network
latent_v_dim = 64  # dimension of value part of q-network
latent_a_dim = 64  # dimension of advantage part of q-network
q_network = comp.q_network.DoubleQNetwork(latent_v_dim, latent_a_dim)

# Policy
epsilon = ParamScheduler(init_value=1, decay_mode="lin", half_life_period=125000, minimum=0.01)

# Miscellaneous
exp_buf_size = 200000  # total number of transitions that can fit into the agent's replay memory
stack_size = 4

agent = Agent(env=env,
              stem_network=stem_model,
              q_network=q_network,
              name="shorter_max_ep_len",
              replay_batch_size=replay_batch_size,
              stack_size=stack_size,
              optimizer=optimizer,
              use_double=use_double,
              use_pretrained=False,
              seed=seed)
# agent.restore("new_fruit_spawning")
# agent.just_play(verbose=True, epsilon=1)

agent.practice(num_parallel_steps=1000000,
               replay_period=replay_period,
               replay_size_multiplier=replay_size_multiplier,
               max_replay_size=max_replay_size,
               replay_epochs=replay_epochs,
               min_hist_len=min_hist_len,
               learning_rate=learning_rate,
               target_sync_period=target_sync_period,
               actor_sync_period=actor_sync_period,
               gamma=gamma,
               epsilon=epsilon,
               use_mc_return=use_mc_return,
               alpha=0.5,
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
