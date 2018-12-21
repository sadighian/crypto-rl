from trading_gym.dqn_agent import DqnAgent


if __name__ == '__main__':
    print('training...')
    agent = DqnAgent(step_size=4, window_length=4, train=True, training_steps=10000, weights=None)
    agent.start()
    print('...done training.')
