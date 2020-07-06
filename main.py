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
                               minibatch=64,  # 32
                               sync_period=1024,
                               gamma=0.99,
                               epsilon=0.8,
                               anneal=0.9999,
                               dueling=True,  # True
                               latent_dim=512,  # 512
                               experience_path="data/tobias.hdf5")

    agent.memory.import_experience("data/experience_backup.hdf5")
    agent.restore_model("models/tobias")
    # agent.learn_from_experience()
    agent.run()
    agent.save_model("models/tobias_2")
    agent.memory.export_experience('data/tobias.hdf5', compress=True)


if __name__ == "__main__":
    main()
