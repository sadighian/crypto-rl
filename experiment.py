from trading_gym.dqn_agent import DqnAgent


if __name__ == '__main__':
    print('Starting training...')
    agent = DqnAgent(step_size=1,
                     window_length=2*60,
                     train=True,
                     max_position=1,
                     weights=False,
                     fitting_file='LTC-USD_20181120.xz',
                     testing_file='LTC-USD_20181121.xz')
    agent.start()
    print('...done training.')
