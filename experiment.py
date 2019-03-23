from trading_gym.agent import Agent


if __name__ == '__main__':
    print('Starting training...')

    kwargs = {
        'step_size': 1,
        'window_size': 20,  # number of shapshots to include in the frame rendered by the CNN
        'train': True,  # if False, episodes start at frame 0
        'max_position': 1,
        'weights': False,  # if True, agent loads saved weights
        'fitting_file': 'ETH-USD_2018-12-31.csv',
        'testing_file': 'ETH-USD_2019-01-01.csv',
        'seed': 1
    }

    agent = Agent(**kwargs)
    agent.start()
    print('...training...')
