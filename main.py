import src.agents.dqn as dqn


def main(*args):
    # Deep Q-Network agent, commented values are used in Nikonova et al.
    agent = dqn.ClientDQNAgent(learning_rate=0.0001,  # 0.0001
                               dueling=True,  # True
                               latent_dim=1024)  # 512

    # agent.memory.import_experience("temp/experience_46000.hdf5")
    agent.restore_model("models/link")
    # agent.learn_from_experience()
    # agent.just_play()
    '''agent.practice(num_episodes=50000,
                   minibatch=128,  # 32
                   sync_period=1024,
                   gamma=0.8,
                   epsilon=1,
                   anneal=0.99995,
                   replay_period=8,
                   grace_factor=0.25)'''
    agent.validate()
    # agent.save_model("models/link")
    # agent.memory.export_experience('data/link_46000_no_grace.hdf5', compress=True)


if __name__ == "__main__":
    main()
