from trading_gym.dqn_agent import DqnAgent


if __name__ == '__main__':
    print('Starting training...')
    agent = DqnAgent(step_size=1, window_length=5, train=True, max_position=1, weights=True)
    agent.start()
    print('...done training.')
