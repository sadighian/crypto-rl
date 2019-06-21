import numpy as np
from datetime import datetime as dt
import gym
from gym_trading.envs.price_jump import PriceJump
import logging
import gym_trading


# logging
logging.basicConfig(level=logging.INFO, format='[%(asctime)s] %(message)s')
logger = logging.getLogger('test_price_jump')


def test_price_jump_python():
    start_time = dt.now()
    config = {
        'training': True,
        'fitting_file': 'ETH-USD_2018-12-31.xz',
        'testing_file': 'ETH-USD_2019-01-01.xz',
        'step_size': 1,
        'max_position': 1,
        'window_size': 5,
        'frame_stack': False
    }
    logger.info('test_price_jump_python() configs are {}'.format(config))

    env = PriceJump(**config)
    total_reward = 0.0

    i = 0
    done = False
    env.reset()

    logger.info('Environment reset. Now starting simulation')

    while not done:
        i += 1
        if i % 200 == 0:
            action = np.random.randint(3)
        else:
            action = 0

        state, reward, done, _ = env.step(action)
        total_reward += reward

        if reward != 0.0:
            logger.debug('reward: %.5f | total_reward %.5f vs. broker: %.5f | '
                         '%.5f  %.5f' %
                         (reward, total_reward, env.broker.get_total_pnl(
                             env.midpoint),
                          env.broker.get_realized_pnl(),
                          env.broker.get_unrealized_pnl(env.midpoint)))

        if done:
            elapsed = (dt.now() - start_time).seconds
            logger.info('Done on step #%i @ %i steps/second' %
                        (i, i*PriceJump.action_repeats // elapsed))
            break

    logger.info('Total reward: %.4f\nTotal pnl: %.4f' %
                (total_reward,
                 env.broker.get_total_pnl(env.midpoint)))


def test_price_jump_gym():
    start_time = dt.now()
    import gym_trading

    config = {
        'training': True,
        'fitting_file': 'ETH-USD_2018-12-31.xz',
        'testing_file': 'ETH-USD_2019-01-01.xz',
        'step_size': 1,
        'max_position': 1,
        'window_size': 5,
        'frame_stack': False
    }
    logger.info('test_price_jump_gym() configs are {}'.format(config))

    env = gym.make(PriceJump.id, **config)
    total_reward = 0.0

    i = 0
    done = False
    env.reset()

    logger.info('Environment reset. Now starting simulation')

    while not done:
        i += 1

        if i % 200 == 0:
            action = np.random.randint(3)
        else:
            action = 0

        state, reward, done, _ = env.step(action)
        total_reward += reward

        # if reward != 0.0:
        #     print('reward = %.4f' % reward)

        if done:
            elapsed = (dt.now() - start_time).seconds
            logger.info('Done on step #%i @ %i steps/second' %
                        (i, i*PriceJump.action_repeats // elapsed))
            break

    logger.info('Total reward: %.4f' % total_reward)


if __name__ == '__main__':
    test_price_jump_python()
    test_price_jump_gym()

