import numpy as np
from datetime import datetime as dt
from trading_gym import TradingGym


if __name__ == '__main__':
    start_time = dt.now()

    params = {
        'training': True,
        'env_id': 'coinbase-bitfinex-v0',
        'step_size': 1,
        'fee': 0.003,
        'max_position': 1,
        'fitting_file': 'BTC-USD_20181120.xz',
        'testing_file': 'BTC-USD_20181121.xz'
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
            env.render()
            break

    print('Total reward: %.4f\nTotal pnl: %.4f' % (total_reward, env.broker.get_total_pnl(env._midpoint)))
    elapsed = (dt.now() - start_time).seconds
    print('\nCompleted in %i seconds' % elapsed)
