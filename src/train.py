from src.agents.dqn import *
from src.envs import *
import src.agents.models as models

# General parameters
num_parallel_envs = 500
replay_period = 64
sync_period = 128  # 256
replay_size_multiplier = 4  # multiplier of 4 means that, per replay on average, 25 % of data is unseen.
batch_size = 1024
replay_epochs = 1

# Stem model
latent_dim = 128
latent_depth = 1  # number of latent (dense) layers (if supported)
stem_model = models.tetris.ClassicConv(latent_dim=latent_dim)

# Q network
latent_v_dim = 64  # dimension of value part of q-network
latent_a_dim = 64  # dimension of advantage part of q-network
q_network = models.q_network.DoubleQNetwork(latent_v_dim, latent_a_dim)

# Learning rate
learning_rate = LearningRate(initial_learning_rate=0.0004,  # lr <= 0.0001 recommended
                             warmup_episodes=0,  # number of episodes for linear LR warm-up (0 for no warm-up)
                             half_life_period=300000)  # determines decay, None for no decay

# Epsilon
epsilon = Epsilon(init_value=1,
                  decay_mode="exp",  # "exp" for exponential, "lin" for linear
                  decay_rate=0.9996,
                  minimum=0)

# Miscellaneous
obs_buf_size = 2000  # number of transitions that can fit into the observations buffer per env, = max episode length +1
exp_buf_size = 4000000  # total number of transitions that can fit into the agent's replay memory

# load_and_play("finnson_improved", Snake)

agent = TFDQNAgent(env_type=Tetris,
                   stem_model=stem_model,
                   q_network=q_network,
                   name="half_latent",
                   num_parallel_envs=num_parallel_envs,  # look table for optimal value
                   use_double=True,
                   obs_buf_size=obs_buf_size,
                   mem_size=exp_buf_size,
                   use_pretrained=False)

# agent.restore("double_lr")

agent.practice(num_parallel_steps=1000000,
               replay_period=replay_period,
               replay_size_multiplier=replay_size_multiplier,
               batch_size=batch_size,
               replay_epochs=replay_epochs,
               learning_rate=learning_rate,
               sync_period=sync_period,
               gamma=0.9995,
               epsilon=epsilon,
               delta=0,
               delta_anneal=1,
               alpha=0.7,
               verbose=False)

'''print("\nTerminals:")
ids = np.where(agent.memory.get_terminals())[0]
if len(ids):
    agent.print_transitions(ids[:100])

print("\nLast 100:")
length = agent.memory.get_length()
agent.print_transitions(range(length - 100, length))'''
