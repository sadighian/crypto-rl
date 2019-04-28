import numpy as np
from datetime import datetime as dt
import gym
from gym_trading.envs.trading_gym import TradingGym


def test_trading_gym():
    start_time = dt.now()
    config = {
        'training': True,
        'fitting_file': 'ETH-USD_2018-12-31.xz',
        'testing_file': 'ETH-USD_2019-01-01.xz',
        'step_size': 1,
        'max_position': 1,
        'window_size': 5,
        'seed': 1,
        'frame_stack': False
    }

    total_reward = 0.0
    env = TradingGym(**config)

    for i in range(env.data.shape[0]):
        if i % 200 == 0:
            action = np.random.randint(3)
        else:
            action = 0

        state, reward, done, _ = env.step(action)
        total_reward += reward

        if reward != 0.0:
            print('reward: %.5f | total_reward %.5f vs. broker: %.5f | %.5f  %.5f' %
                  (reward, total_reward, env.broker.get_total_pnl(env.midpoint),
                   env.broker.get_realized_pnl(), env.broker.get_unrealized_pnl(env.midpoint)))
            print('midpoint: %.4f | avg.long: %.4f | avg.short: %.4f' %
                  (env.midpoint, env.broker.long_inventory.average_price,
                   env.broker.short_inventory.average_price))
            print("")

        if done:
            elapsed = (dt.now() - start_time).seconds
            print('Done on step #%i @ %i ticks/second' % (i, i // elapsed))
            break

    print('Total reward: %.4f\nTotal pnl: %.4f' %
          (total_reward,
           env.broker.get_total_pnl(env.midpoint)))


def test_gym():
    start_time = dt.now()
    import gym_trading

    config = {
        'training': True,
        'fitting_file': 'ETH-USD_2018-12-31.xz',
        'testing_file': 'ETH-USD_2019-01-01.xz',
        'step_size': 1,
        'max_position': 1,
        'window_size': 5,
        'seed': 1,
        'frame_stack': False
    }

    env = gym.make(TradingGym.id, **config)
    total_reward = 0.0

    i = 0
    done = False
    env.reset()

    while not done:
        i += 1

        if i % 200 == 0:
            action = np.random.randint(3)
        else:
            action = 0

        state, reward, done, _ = env.step(action)
        total_reward += reward

        if reward != 0.0:
            print('reward = %.4f' % reward)

        if done:
            elapsed = (dt.now() - start_time).seconds
            print('Done on step #%i @ %i ticks/second' % (i, i // elapsed))
            break

    print('Total reward: %.4f' % total_reward)


if __name__ == '__main__':
    test_trading_gym()
    test_gym()

