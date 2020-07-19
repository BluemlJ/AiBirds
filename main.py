import src.agents.dqn as dqn


def main(*args):
    # Deep Q-Network agent, commented values are used in Nikonova et al.
    agent = dqn.ClientDQNAgent(name="test_E",
                               learning_rate=0.0001,  # 0.0001
                               dueling=True,  # True
                               latent_dim=512)  # 512

    # agent.restore_experience()
    # agent.restore_model()
    # agent.learn_from_experience()
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
                   delta_anneal=0.99995)
    # agent.validate()
    # agent.save_model()
    # agent.save_experience(compress=True)


if __name__ == "__main__":
    main()
