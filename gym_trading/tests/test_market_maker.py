import unittest
import numpy as np
from datetime import datetime as dt
import gym
from gym_trading.envs.market_maker import MarketMaker
import gym_trading


class MyTestCase(unittest.TestCase):

    def test_market_maker_gym(self):
        start_time = dt.now()

        config = {
            'training': False,
            'fitting_file': 'ETH-USD_2018-12-31.xz',
            'testing_file': 'ETH-USD_2019-01-01.xz',
            'step_size': 1,
            'max_position': 5,
            'window_size': 20,
            'seed': 1,
            'action_repeats': 10,
            'format_3d': False,
            'z_score': False,
        }

        env = gym.make(MarketMaker.id, **config)
        total_reward = 0.0

        i = 0
        done = False
        env.reset()

        while not done:
            i += 1

            if i % 30000 == 0:
                action = np.random.randint(env.action_space.n)
                break
            else:
                action = 0

            state, reward, done, _ = env.step(action)
            total_reward += reward
            # env.render()

            if done:
                elapsed = (dt.now() - start_time).seconds
                print('Done on step #%i @ %i steps/second' % (i, (i // elapsed) *
                      env.action_repeats))
                break

        env.reset()
        print('Total reward: %.4f' % total_reward)
        self.assertEqual(True, done)


if __name__ == '__main__':
    unittest.main()
