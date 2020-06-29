import src.agents.naive as na
import src.agents.dummy as dum
import src.agents.dqn as dqn


def main(*args):
    # agent = na.ClientNaiveAgent()  # Provided naive agent
    # agent = dum.ClientDummyAgent(start_level=1999)  # Dummy NN agent

    # Deep Q-Network agent
    agent = dqn.ClientDQNAgent(start_level=1,
                               num_episodes=7000,
                               sim_speed=60,
                               replay_period=8,
                               learning_rate=0.0001,
                               minibatch=32,
                               sync_period=1024,
                               gamma=0.99,
                               epsilon=1,
                               anneal=0.9999)

    # agent.learn_from_experience("data/experiences.bz2")
    # agent.restore_model("models/justus")
    agent.run()
    agent.save_model("models/kevin")

    agent2 = dqn.ClientDQNAgent(start_level=1,
                               num_episodes=7000,
                               sim_speed=60,
                               replay_period=8,
                               learning_rate=0.0001,
                               minibatch=32,
                               sync_period=1024,
                               gamma=0.99,
                               epsilon=1,
                               anneal=0.9999)

    # agent.learn_from_experience("data/experiences.bz2")
    # agent.restore_model("models/justus")
    agent2.run()
    agent2.save_model("models/volker")


if __name__ == "__main__":
    main()
