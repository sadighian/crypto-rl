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


# DEFAULT_ACTION_SET = (
#     (1, 0, 0, 0, 0),  # 0. Buy
#     (0, 1, 0, 0, 0),  # 1. Close-buy
#     (0, 0, 1, 0, 0),  # 2. Short
#     (0, 0, 0, 1, 0),  # 3. Close-short
#     (0, 0, 0, 0, 1),  # 4. Do nothing
# )
DEFAULT_ACTION_SET = {
    0: 'buy',
    1: 'close-buy',
    2: 'short',
    3: 'close-short',
    4: 'no-action'
}


EXTRA_FEATURES = ['long_inventory', 'short_inventory']


class TradingGym(Env):

    def __init__(self, query, lags, training=True, env_id='coinbasepro-bitfinex-v0', step_size=1,
                 fee=0.003, max_position=1):
        # properties required for instantiation
        self.training = training
        self.env_id = env_id
        self.step_size = step_size
        self.fee = fee
        self.max_position = max_position

        # properties that get reset()
        self.reward = None
        self.done = False
        self.local_step_number = 0

        # get historical data for simulations
        self.broker = Broker(ccy=query['ccy'][0], max_position=max_position)
        self.sim = Sim()
        self.features = self.sim.get_feature_labels(include_system_time=False, lags=lags)
        self.midpoint_index = len(self.sim.get_feature_labels(include_system_time=False)) + 2
        print('midpoint index is %i' % self.midpoint_index)
        # self.data = self.sim.get_env_data(query=query,
        #                                   coinbaseOrderBook=CoinbaseOrderBook(query['ccy'][0]),
        #                                   bitfinexOrderBook=BitfinexOrderBook(query['ccy'][1]),
        #                                   lags=lags)
        self.data = pd.read_csv(os.getcwd() + '/env_data_history.csv')
        print('...loaded data from csv.')
        delete_columns = [self.data.columns[0], 't']
        print('...deleting column: %s' % delete_columns)
        del self.data[self.data.columns[0]]
        del self.data['t']

        self.data = self.data.values
        self.observation = self.reset()

        # derive gym.env properties
        self.actions = DEFAULT_ACTION_SET
        self.action_space = spaces.Discrete(len(self.actions))
        self.observation_space = spaces.Box(low=-np.inf,
                                            high=np.inf,
                                            shape=(1, len(self.features) + len(EXTRA_FEATURES)),
                                            dtype=np.float32)
        self.env_id = env_id

        # logging
        logging.basicConfig(level=logging.INFO, format='[%(asctime)s] %(message)s')
        self.logger = logging.getLogger(env_id)
        self.logger.info('Making new env: {}'.format(env_id))

    def step(self, action):
        if self.done:
            self.observation = self.reset()
            return self.observation, self.reward, self.done, {'isDone': self.done}

        self.observation = self.data[self.local_step_number]
        self.reward = self.get_reward(action, self.observation)
        self.observation = np.concatenate((self.observation, self.create_position_features()))
        self.observation = self.observation.reshape(1, -1)

        self.local_step_number += self.step_size
        if self.local_step_number > self.data.shape[0] - 2:
            self.done = True

        return self.observation, self.reward, self.done, {'local_step_number': self.local_step_number}

    def reset(self):
        self.reward = None
        self.done = False
        self.broker.reset()
        self.local_step_number = random.randint(0, self.data.shape[0] - int(self.data.shape[0] * .2)) \
            if self.training else 0
        self.observation = np.concatenate((self.data[self.local_step_number], self.create_position_features()))
        self.observation = self.observation.reshape(1, -1)
        self.local_step_number += 1
        return self.observation

    def render(self, mode='human'):
        return None

    def close(self):
        return

    def seed(self, seed=None):
        np_random, seed = seeding.np_random(seed)
        return [seed]

    def get_reward(self, action, observation):
        midpoint = observation[self.midpoint_index]

        if action == 0:  # buy
            order = {
                'price': midpoint + (midpoint * self.fee),
                'size': 10000.0,
                'side': 'long'
            }
            self.broker.add(order=order)
        elif action == 1:  # close-buy
            order = {
                'price': midpoint,
                'size': 10000.0,
                'side': 'long'
            }
            self.broker.remove(order=order)
        elif action == 2:  # short
            order = {
                'price': midpoint - (midpoint * self.fee),
                'size': 10000.0,
                'side': 'short'
            }
            self.broker.add(order=order)
        elif action == 3:  # cover-short
            order = {
                'price': midpoint,
                'size': 10000.0,
                'side': 'short'
            }
            self.broker.remove(order=order)
        elif action == 4:  # do nothing
            pass
        else:
            print('Unknown action to take in get_reward(): action={} | midpoint={}'.format(action, midpoint))

        unrealized_pnl = self.broker.get_unrealized_pnl(midpoint=midpoint)
        realized_pnl = self.broker.get_realized_pnl()
        #  TODO: Add penalty for not having an empty inventory & short borrowing fee
        return unrealized_pnl + realized_pnl

    def create_position_features(self):
        return self.broker.long_inventory.position_count / self.max_position, \
               self.broker.short_inventory.position_count / self.max_position
