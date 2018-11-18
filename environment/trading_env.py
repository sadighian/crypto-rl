import gym
from simulator import Simulator as sim
from coinbase_connector.coinbase_orderbook import CoinbaseOrderBook
from bitfinex_connector.bitfinex_orderbook import BitfinexOrderBook
from configurations.configs import TIMEZONE
from environment.broker import Broker
from environment.position import Position


import os
import logging

import pandas as pd
import numpy as np
from datetime import datetime as dt

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as patches


DEFAULT_ACTION_SET = (
    (1, 0, 0, 0, 0),  # 0. Buy
    (0, 1, 0, 0, 0),  # 1. Close-buy
    (0, 0, 1, 0, 0),  # 2. Short
    (0, 0, 0, 1, 0),  # 3. Close-short
    (0, 0, 0, 0, 1),  # 4. Do nothing
)


class TradingEnv(gym.Env):

    # def __init__(self, query, lags=5):
        # self.query = query
        # self.lags = lags
        # self.sim = sim()
        # self.data = self.sim.get_env_data(query=self.query, lags=self.lags).values
        # self.n_features = self.data.shape[1]
        # self.action_space = gym.spaces.Discrete(n=DEFAULT_ACTION_SET)
        # self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=self.data.shape, dtype=np.float32)
        # self.shape = (self.lags, self.n_features)
        # self.local_step_number = int(0)
        # self.observation = None
        # self.reward = None
        # self.done = False
    def __init__(self, env_id, step_size, fee, max_position=5, fluc_div=100.0, gameover_limit=5):
        """
        #assert df
        # need deal price as essential and specified the df format
        # obs_data_leng -> observation data length
        # step_len -> when call step rolling windows will + step_len
        # df -> dataframe that contain data for trading(format as...)
            # price
            # datetime
            # serial_number -> serial num of deal at each day recalculating

        # fee -> when each deal will pay the fee, set with your product
        # max_position -> the max market position for you trading share
        # deal_col_name -> the column name for cucalate reward used.
        # feature_names -> list contain the feature columns to use in trading status.
        # ?day trade option set as default if don't use this need modify
        """
        self.sim = sim()
        lags = 0
        query = {
            'ccy': ['BCH-USD', 'tBCHUSD'],
            'start_date': 20181110,
            'end_date': 20181113
        }

        data_normalized = self.sim.get_env_data(query=query,
                                                coinbaseOrderBook=CoinbaseOrderBook(query['ccy'][0]),
                                                bitfinexOrderBook=BitfinexOrderBook(query['ccy'][1]),
                                                lags=lags)
        self.data = data_normalized[self.sim.get_feature_labels(include_system_time=False, lags=lags)]
        self.observation = None
        self.reward = None

        self.broker = Position(ccy=query['ccy'][0], max_position=1)

        self.action_space = np.array([5, ])
        self.gym_actions = range(5)

        self.step_size = step_size
        self.fee = fee
        self.max_position = max_position

        self.fluc_div = fluc_div
        self.gameover = gameover_limit

        self.render_on = False
        self.done = False
        self.local_step_number = 0

        logging.basicConfig(level=logging.INFO, format='[%(asctime)s] %(message)s')
        self.logger = logging.getLogger(env_id)
        self.logger.info('Making new env: {}'.format(env_id))

    def step(self, action):
        """Run one timestep of the environment's dynamics. When end of
        episode is reached, you are responsible for calling `reset()`
        to reset this environment's state.

        Accepts an action and returns a tuple (observation, reward, done, info).

        Args:
            action (object): an action provided by the environment

        Returns:
            observation (object): agent's observation of the current environment
            reward (float) :amount of reward returned after previous action
            done (boolean): whether the episode has ended, in which case further step() calls
                            will return undefined results
            info (dict):    contains auxiliary diagnostic information (helpful for debugging,
                            and sometimes learning)
        """

        if self.done:
            self.reset()
            return self.observation, self.reward, self.done, {'isDone': self.done}

        self.observation = self.data[self.local_step_number]

        self.reward = self.get_reward(action, self.observation)

        self.local_step_number += self.step_size

        return self.observation, self.reward, self.done, {'local_step_number': self.local_step_number}

    def reset(self):
        """Resets the state of the environment and returns an initial observation.

        Returns: observation (object): the initial observation of the
            space.
        """
        return

    def render(self, mode='human'):
        """Renders the environment.

        The set of supported modes varies per environment. (And some
        environments do not support rendering at all.) By convention,
        if mode is:

        - human: render to the current display or terminal and
          return nothing. Usually for human consumption.
        - rgb_array: Return an numpy.ndarray with shape (x, y, 3),
          representing RGB values for an x-by-y pixel image, suitable
          for turning into a video.
        - ansi: Return a string (str) or StringIO.StringIO containing a
          terminal-style text representation. The text can include newlines
          and ANSI escape sequences (e.g. for colors).

        Note:
            Make sure that your class's metadata 'render.modes' key includes
              the list of supported modes. It's recommended to call super()
              in implementations to use the functionality of this method.

        Args:
            mode (str): the mode to render with
            close (bool): close all open renderings

        Example:

        class MyEnv(Env):
            metadata = {'render.modes': ['human', 'rgb_array']}

            def render(self, mode='human'):
                if mode == 'rgb_array':
                    return np.array(...) # return RGB frame suitable for video
                elif mode is 'human':
                    ... # pop up a window and render
                else:
                    super(MyEnv, self).render(mode=mode) # just raise an exception
        """
        return None

    def close(self):
        """Override _close in your subclass to perform any necessary cleanup.

        Environments will automatically close() themselves when
        garbage collected or when the program exits.
        """
        return

    def seed(self, seed=None):
        """Sets the seed for this env's random number generator(s).

        Note:
            Some environments use multiple pseudorandom number generators.
            We want to capture all such seeds used in order to ensure that
            there aren't accidental correlations between multiple generators.

        Returns:
            list<bigint>: Returns the list of seeds used in this env's random
              number generators. The first value in the list should be the
              "main" seed, or the value which a reproducer should pass to
              'seed'. Often, the main seed equals the provided 'seed', but
              this won't be true if seed=None, for example.
        """
        self.np_random, seed = gym.utils.np_random(seed)
        return [seed]

    def get_reward(self, action, observation):
        midpoint = observation['m']

        if action == 0:
            # buy
            order = {
                'price': midpoint + (midpoint*self.fee),
                'size': 10000.0,
                'side': 'long'
            }
            self.broker.add(order=order)
        elif action == 1:
            # close-buy
            order = {
                'price': midpoint,
                'size': 10000.0,
                'side': 'long'
            }
            self.broker.remove(order=order)
        elif action == 2:
            # short
            order = {
                'price': midpoint - (midpoint*self.fee),
                'size': 10000.0,
                'side': 'short'
            }
            self.broker.add(order=order)
        elif action == 3:
            # cover-short
            order = {
                'price': midpoint,
                'size': 10000.0,
                'side': 'short'
            }
            self.broker.remove(order=order)
        elif action == 4:
            # do nothing
            pass
        else:
            print('Unknown action to take in get_reward(): action={} | midpoint={}'.format(action, midpoint))

        unrealized_pnl = self.broker.get_unrealized_pnl(midpoint=midpoint)
        realized_pnl = self.broker.get_realized_pnl()
        #  TODO: Add penalty for not having an empty inventory & short borrowing fee
        return unrealized_pnl + realized_pnl
