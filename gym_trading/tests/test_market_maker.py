import numpy as np
from datetime import datetime as dt
import gym
from gym_trading.envs.market_maker import MarketMaker
import gym_trading


def test_gym():
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

    env = gym.make(MarketMaker.id, **config)
    total_reward = 0.0

    i = 0
    done = False
    env.reset()

    while not done:
        i += 1

        if i % 200 == 0:
            action = np.random.randint(env.action_space.n)
        else:
            action = 0

        state, reward, done, _ = env.step(action)
        total_reward += reward

        # if reward != 0.0:
        #     print('reward = %.4f' % reward)

        if done:
            elapsed = (dt.now() - start_time).seconds
            print('Done on step #%i @ %i ticks/second' % (i, i // elapsed))
            break

    print('Total reward: %.4f' % total_reward)


def test_market_maker():
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

    env = gym.make(MarketMaker.id, **config)

    env.reset()
    for _ in range(100):
        state, reward, done, _ = env.step(0)

    state, reward, done, _ = env.step(13)

    for _ in range(6000):
        state, reward, done, _ = env.step(3)
        if reward > 0.0001:
            print('reward={}'.format(reward))
            break

    state, reward, done, _ = env.step(13)
    for _ in range(10000):
        state, reward, done, _ = env.step(0)

    elapsed = (dt.now() - start_time).seconds
    print('Done in %i seconds' % elapsed)


if __name__ == '__main__':
    test_gym()
    # test_market_maker()

