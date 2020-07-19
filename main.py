import src.agents.dqn as dqn
import os
import plaidml.keras  # for AMD GPU acceleration
plaidml.keras.install_backend()
os.environ["KERAS_BACKEND"] = "plaidml.keras.backend"


def main(*args):
    # Deep Q-Network agent, commented values are used in Nikonova et al.
    agent = dqn.ClientDQNAgent(name="peter",
                               learning_rate=0.0001,  # 0.0001
                               dueling=True,  # True
                               latent_dim=512)  # 512

    # agent.restore_experience("data/experience_1000.hdf5")
    # agent.restore_model()
    """agent.learn_from_experience(num_epochs=1000,
                                minibatch=128,  # 32
                                sync_period=125,
                                gamma=0.9,  # 0.99
                                grace_factor=0.25,  # 0
                                delta=1)  # 0"""
    # agent.just_play()
    agent.practice(num_episodes=50000,
                   minibatch=128,  # 32
                   sync_period=1024,
                   gamma=0.9,  # 0.99
                   epsilon=1,
                   epsilon_anneal=0.9999,
                   replay_period=8,
                   grace_factor=0.25,  # 0
                   delta=0,  # 0
                   delta_anneal=0.995)
    agent.validate()
    agent.save_model()
    agent.save_experience(compress=True)


if __name__ == "__main__":
    main()
