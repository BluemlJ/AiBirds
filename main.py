import src.agents.naive as na
import src.agents.dummy as dum
import src.agents.dqn as dqn


def main(*args):
    # agent = na.ClientNaiveAgent()  # Provided naive agent
    # agent = dum.ClientDummyAgent(start_level=1999)  # Dummy NN agent

    # Deep Q-Network agent
    agent = dqn.ClientDQNAgent(start_level=1,
                               num_episodes=10000,
                               sim_speed=60,
                               replay_period=8,
                               learning_rate=0.0001,  # 0.0001 Nikonova et al.
                               minibatch=32,
                               sync_period=1024,
                               gamma=0.99,
                               epsilon=1,
                               anneal=0.9999,
                               dueling=True,
                               latent_dim=512)  # 512 Nikonova et al.

    # agent.learn_from_experience("data/experiences.bz2")
    # agent.restore_model("models/justus")
    agent.run()
    agent.save_model("models/nadine")


if __name__ == "__main__":
    main()
