import numpy as np
from datetime import datetime as dt
import gym
from gym_trading.envs.market_maker import MarketMaker
import gym_trading
import logging


# logging
logging.basicConfig(level=logging.INFO, format='[%(asctime)s] %(message)s')
logger = logging.getLogger('test_market_maker')


def test_market_maker_gym():
    start_time = dt.now()

    config = {
        'training': False,
        'fitting_file': 'ETH-USD_2018-12-31.xz',
        'testing_file': 'ETH-USD_2019-01-01.xz',
        'step_size': 1,
        'max_position': 5,
        'window_size': 1,
        'seed': 1,
        'action_repeats': 10,
        'frame_stack': False
    }

    env = gym.make(MarketMaker.id, **config)
    total_reward = 0.0

    i = 0
    done = False
    env.reset()

    while not done:
        i += 1

        if i % 3 == 0:
            action = np.random.randint(env.action_space.n)
        else:
            action = 0

        state, reward, done, _ = env.step(action)
        total_reward += reward
        # env.render()

        # if abs(reward) >= 0.01:
        #     print('reward = %.4f' % reward)

        if done:
            elapsed = (dt.now() - start_time).seconds
            print('Done on step #%i @ %i ticks/second' % (i, (i // elapsed) *
                  MarketMaker.action_repeats))
            break

    env.reset()
    print('Total reward: %.4f' % total_reward)


def test_market_maker_python():
    start_time = dt.now()

    config = {
        'training': True,
        'fitting_file': 'ETH-USD_2018-12-31.xz',
        'testing_file': 'ETH-USD_2019-01-01.xz',
        'step_size': 1,
        'max_position': 1,
        'window_size': 5,
        'seed': 1,
        'action_repeats': 10,
        'frame_stack': False
    }
    logger.info('test_market_maker_python() configs are {}'.format(config))

    env = gym.make(MarketMaker.id, **config)
    env.reset()

    logger.info('Taking no action for 100 steps')
    for _ in range(100):
        state, reward, done, _ = env.step(0)

    logger.info('Doing action 13 for 1 step')
    state, reward, done, _ = env.step(13)

    logger.info('Doing action 3 for 6000 steps')
    for _ in range(6000):
        state, reward, done, _ = env.step(3)
        if reward > 0.0001:
            logger.info('reward={}'.format(reward))
            break

    logger.info('Doing action 13 for 1 step')
    state, reward, done, _ = env.step(13)

    logger.info('Taking no action for 10000 steps')
    for _ in range(10000):
        state, reward, done, _ = env.step(0)

    elapsed = (dt.now() - start_time).seconds
    logger.info('Done in %i seconds' % elapsed)


if __name__ == '__main__':
    test_market_maker_gym()
    test_market_maker_python()
