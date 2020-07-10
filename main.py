import src.agents.naive as na
import src.agents.dummy as dum
import src.agents.dqn as dqn


def main(*args):
    # agent = na.ClientNaiveAgent()  # Provided naive agent
    # agent = dum.ClientDummyAgent(start_level=1999)  # Dummy NN agent

    # Deep Q-Network agent, commented values are used in Nikonova et al.
    agent = dqn.ClientDQNAgent(start_level=1,
                               num_episodes=20000,
                               sim_speed=60,
                               replay_period=8,
                               learning_rate=0.0001,  # 0.0001
                               minibatch=128,  # 32
                               sync_period=1024,
                               gamma=0.99,
                               epsilon=0.5,
                               anneal=0.9999,
                               dueling=True,  # True
                               latent_dim=512,  # 512
                               experience_path="temp/test.hdf5")

    agent.memory.import_experience("data/tobias_10000_no_grace_2.hdf5")
    agent.restore_model("models/tobias_4")
    # agent.learn_from_experience()
    agent.run()
    agent.save_model("models/tobias_5")
    agent.memory.export_experience('data/tobias_20000_no_grace_3.hdf5', compress=True)


if __name__ == "__main__":
    main()
