import numpy as np
from datetime import datetime as dt
from trading_gym import TradingGym


if __name__ == '__main__':
    start_time = dt.now()

    config = {
        'training': True,
        'fitting_file': 'ETH-USD_2018-12-31.csv',
        'testing_file': 'ETH-USD_2019-01-01.csv',
        'env_id': 'coinbase-bitfinex-v1',
        'step_size': 1,
        'fee': 0.003,
        'max_position': 1,
        'window_size': 5,
        'seed': 13
    }

    total_reward = 0.0
    env = TradingGym(**config)

    for i in range(env.data.shape[0]):
        if i % 200 == 0:
            action = np.random.randint(3)
        else:
            action = 0

        state, reward, done = env.step(action)

        total_reward += reward

        if reward != 0.0:
            print('reward: %.5f | total_reward %.5f vs. broker: %.5f | %.5f  %.5f' %
                  (reward, total_reward, env.broker.get_total_pnl(env._midpoint),
                  env.broker.get_realized_pnl(), env.broker.get_unrealized_pnl(env._midpoint)))
            print('midpoint: %.4f | avg.long: %.4f | avg.short: %.4f' %
                  (env._midpoint, env.broker.long_inventory.average_price,
                   env.broker.short_inventory.average_price))
            print("")

        if done:
            elapsed = (dt.now() - start_time).seconds
            print('Done on step #%i @ %i ticks/second' % (i, i//elapsed))
            break

    print('Total reward: %.4f\nTotal pnl: %.4f' %
          (total_reward,
           env.broker.get_total_pnl(env._midpoint)))
