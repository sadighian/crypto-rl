import numpy as np
import pandas as pd
from datetime import datetime as dt
from trading_gym import TradingGym


if __name__ == '__main__':
    start_time = dt.now()

    params = {
        'training': True,
        'env_id': 'coinbase-bitfinex-v0',
        'step_size': 1,
        'fee': 0.006,
        'max_position': 1
    }
    total_reward = 0.0
    env = TradingGym(**params)

    for i in range(env.data.shape[0] - 5):
        if i % 2000 == 0:
            action = np.random.randint(3)
        else:
            action = 0
        state, reward, done, info = env.step(action)

        total_reward += reward

        if done:
            print('Done on %i step' % i)
            break

    print('Total reward: %f' % total_reward)
    elapsed = (dt.now() - start_time).seconds
    print('\nCompleted in %i seconds' % elapsed)
