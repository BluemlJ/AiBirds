import src.agents.naive as na
import src.agents.dummy as dum
import src.agents.dqn as dqn


def main(*args):
    # agent = na.ClientNaiveAgent()  # Provided naive agent
    # agent = dum.ClientDummyAgent(start_level=1999)  # Dummy NN agent
    agent = dqn.ClientDQNAgent(sim_speed=50)  # Deep Q-Network agent

    agent.run()


if __name__ == "__main__":
    main()
