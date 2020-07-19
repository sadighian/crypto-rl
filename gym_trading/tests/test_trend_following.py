import unittest
from datetime import datetime as dt

import gym
import numpy as np

from gym_trading.envs.trend_following import TrendFollowing
from gym_trading.utils.decorator import print_time


class TrendFollowingTestCases(unittest.TestCase):

    @print_time
    def test_trend_following_default_reward(self):
        start_time = dt.now()

        config = dict(
            symbol='LTC-USD',
            fitting_file='demo_LTC-USD_20190926.csv.xz',
            testing_file='demo_LTC-USD_20190926.csv.xz',
            max_position=10,
            window_size=1,
            seed=1,
            action_repeats=5,
            training=True,
            format_3d=False,
            reward_type='default',
            ema_alpha=None
        )
        print("**********\n{}\n**********".format(config))

        env = gym.make(TrendFollowing.id, **config)
        reward_list = []

        i = 0
        done = False
        env.reset()

        while not done:
            i += 1

            if i % 300 == 0:
                action = np.random.randint(env.action_space.n)
            else:
                action = 0

            state, reward, done, _ = env.step(action)
            reward_list.append(reward)

            if i > 10000:
                print('Exiting early at step #{} to save time.'.format(i))
                done = True

            if done:
                elapsed = max((dt.now() - start_time).seconds, 1)
                print('Done on step #%i @ %i steps/second' % (
                    i, (i // elapsed) * env.action_repeats))
                print("Max reward: {}\nMin reward: {}".format(max(reward_list),
                                                              min(reward_list)))
                # Visualize results
                env.env.plot_trade_history()
                env.env.plot_observation_history()
                break

        env.reset()
        env.close()
        self.assertEqual(True, done)

    @print_time
    def test_trend_following_differential_sharpe_ratio_reward(self):
        start_time = dt.now()

        config = dict(
            symbol='LTC-USD',
            fitting_file='demo_LTC-USD_20190926.csv.xz',
            testing_file='demo_LTC-USD_20190926.csv.xz',
            max_position=10,
            window_size=1,
            seed=1,
            action_repeats=5,
            training=True,
            format_3d=False,
            reward_type='differential_sharpe_ratio',
            ema_alpha=None
        )
        print("**********\n{}\n**********".format(config))

        env = gym.make(TrendFollowing.id, **config)
        reward_list = []

        i = 0
        done = False
        env.reset()

        while not done:
            i += 1

            if i % 300 == 0:
                action = np.random.randint(env.action_space.n)
            else:
                action = 0

            state, reward, done, _ = env.step(action)
            reward_list.append(reward)

            if i > 10000:
                print('Exiting early at step #{} to save time.'.format(i))
                done = True

            if done:
                elapsed = max((dt.now() - start_time).seconds, 1)
                print('Done on step #%i @ %i steps/second' % (
                    i, (i // elapsed) * env.action_repeats))
                print("Max reward: {}\nMin reward: {}".format(max(reward_list),
                                                              min(reward_list)))
                break

        env.env.plot_observation_history()
        env.reset()
        env.close()
        self.assertEqual(True, done)


if __name__ == '__main__':
    unittest.main()
