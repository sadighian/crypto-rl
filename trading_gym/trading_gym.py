from gym import Env, spaces
from gym.utils import seeding
import random
from simulator import Simulator as Sim
from broker import Broker
from coinbase_connector.coinbase_orderbook import CoinbaseOrderBook
from bitfinex_connector.bitfinex_orderbook import BitfinexOrderBook
import logging
import numpy as np
import pandas as pd
import os
from sklearn.preprocessing import MinMaxScaler


class TradingGym(Env):

    instance_count = 0

    def __init__(self, data,
                 training=True, env_id='coinbasepro-bitfinex-v0',
                 step_size=1, fee=0.003, max_position=1):

        # properties required for instantiation
        self.training = training
        self.env_id = env_id
        self.step_size = step_size
        self.fee = fee
        self.max_position = max_position
        self.inventory_features = ['long_inventory', 'short_inventory']
        self.scaler = MinMaxScaler()

        # properties that get reset()
        self._reward = None
        self._done = False
        self._local_step_number = 0
        self._state = None
        self._next_state = None
        # derive gym.env properties
        self._actions = {
            0: 'buy',
            1: 'close-buy',
            2: 'short',
            3: 'close-short',
            4: 'no-action'
        }
        self._midpoint = None

        # get historical data for simulations
        self.broker = Broker()
        self.sim = Sim()
        self.features = self.sim.get_feature_labels(include_system_time=False, lags=0)

        # load the data
        self.data = data.values
        self.observation = self.reset()

        self.action_space = spaces.Discrete(len(self._actions))
        self.observation_space = spaces.Box(low=-np.inf,
                                            high=np.inf,
                                            shape=(1, len(self.features) + len(self.inventory_features)),
                                            dtype=np.float32)

        # logging
        logging.basicConfig(level=logging.INFO, format='[%(asctime)s] %(message)s')
        self.logger = logging.getLogger(env_id)
        self.logger.info('Making new env: {}'.format(env_id))
        TradingGym.instance_count += 1

    def step(self, action):
        if self._done:
            self.logger.info('{} is done.'.format(self.env_id))
            self.observation = self.reset()
            return self.observation[1], self._reward, self._done, {}

        self._next_state = np.concatenate((self.process_data(self.data[self._local_step_number]),
                                           self.create_position_features()))
        self.observation = (self._state, self._next_state)
        self._state = self._next_state

        self._midpoint = self.data[self._local_step_number][0]
        self._reward = self.get_reward(action)

        self._local_step_number += self.step_size
        if self._local_step_number > self.data.shape[0] - 2:
            self._done = True

        return self.observation[1], self._reward, self._done, {}

    def reset(self):
        print('{} has reset'.format(self.env_name))
        self._reward = None
        self._done = False
        self.broker.reset()
        self._local_step_number = 0
        self._state = np.concatenate((self.process_data(self.data[self._local_step_number]),
                                      self.create_position_features()))
        self._local_step_number += 1
        self._next_state = np.concatenate((self.process_data(self.data[self._local_step_number]),
                                           self.create_position_features()))
        self._local_step_number += 1
        self.observation = (self._state, self._next_state)
        return self.observation

    def render(self, mode='human'):
        return None

    def close(self):
        self.data = None
        self.broker = None
        self.sim = None
        return

    def seed(self, seed=None):
        np_random, seed = seeding.np_random(seed)
        return [seed]

    def get_reward(self, action):
        if action == 0:  # buy
            order = {
                'price': self._midpoint + (self._midpoint * self.fee),
                'size': 10000.0,
                'side': 'long'
            }
            self.broker.add(order=order)
        elif action == 1:  # close-buy
            order = {
                'price': self._midpoint,
                'size': 10000.0,
                'side': 'long'
            }
            self.broker.remove(order=order)
        elif action == 2:  # short
            order = {
                'price': self._midpoint - (self._midpoint * self.fee),
                'size': 10000.0,
                'side': 'short'
            }
            self.broker.add(order=order)
        elif action == 3:  # cover-short
            order = {
                'price': self._midpoint,
                'size': 10000.0,
                'side': 'short'
            }
            self.broker.remove(order=order)
        elif action == 4:  # do nothing
            pass
        else:
            print('Unknown action to take in get_reward(): action={} | midpoint={}'.format(action, self._midpoint))

        # unrealized_pnl = self.broker.get_unrealized_pnl(midpoint=self._midpoint)
        # realized_pnl = self.broker.get_realized_pnl()
        # penalty = self._calculate_penalty(action=action)
        # total_pnl = unrealized_pnl + realized_pnl - penalty
        #
        # _reward = total_pnl - self.prev_total_pnl
        # self.prev_total_pnl = total_pnl + penalty
        return None

    def _calculate_penalty(self, action=0, penalty=0.0):
        _penalty = penalty
        is_long_inventory_full = self.broker.long_inventory.full_inventory
        is_short_inventory_full = self.broker.short_inventory.full_inventory

        if action == 0:  # buy
            if is_long_inventory_full:
                _penalty += self.fee
        elif action == 1:  # close-buy
            pass
        elif action == 2:  # short
            if is_short_inventory_full:
                _penalty += self.fee
        elif action == 3:  # cover-short
            pass
        elif action == 4:  # do nothing
            pass
        else:
            print('Unknown action to take in get_reward(): action={} | midpoint={}'
                  .format(action, self._midpoint))

        return _penalty

    def create_position_features(self):
        return np.array((self.broker.long_inventory.position_count / self.max_position,
                         self.broker.short_inventory.position_count / self.max_position))

    def process_data(self, _next_state):
        return self.scaler.transform(_next_state.reshape(1, -1)).reshape(_next_state.shape)
