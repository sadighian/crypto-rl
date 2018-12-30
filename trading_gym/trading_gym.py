from gym import Env, spaces
from gym.utils import seeding
from .simulator import Simulator as Sim
from .broker import Broker, Order
import logging
import numpy as np
import os

import pandas as pd
import matplotlib.pyplot as plt


class TradingGym(Env):

    def __init__(self, training=True,
                 fitting_file='LTC-USD_20181120.xz',
                 testing_file='LTC-USD_20181121.xz',
                 env_id='coinbasepro-bitfinex-v0',
                 step_size=1,
                 fee=0.003,
                 max_position=1):

        # properties required for instantiation
        self.training = training
        self.env_id = env_id
        self.step_size = step_size
        self.fee = fee
        self.max_position = max_position
        self.inventory_features = ['long_inventory', 'short_inventory', 'long_unrealized_pnl', 'short_unrealized_pnl']
        self._action = 0
        # derive gym.env properties
        self.actions = {
            0: (1, 0, 0),  # 0. do nothing
            1: (0, 1, 0),  # 1. buy
            2: (0, 0, 1)  # 2. sell
        }
        self.sym = fitting_file[:7]  # slice the CCY from the filename
        self.data_for_render = []

        # logging
        logging.basicConfig(level=logging.INFO, format='[%(asctime)s] %(message)s')
        self.logger = logging.getLogger(env_id)
        self.logger.info('Making new env: {}'.format(env_id))

        # properties that get reset()
        self.reward = 0.0
        self.done = False
        self.local_step_number = 0
        self._state = None
        self._midpoint = 0.0

        # get historical data for simulations
        self.broker = Broker(max_position=max_position)
        self.sim = Sim(use_arctic=False)
        self.features = self.sim.get_feature_labels(include_system_time=False)

        # cwd = os.getcwd()
        cwd = os.path.dirname(os.path.realpath(__file__))
        fitting_data_filepath = cwd + '/data_exports/{}'.format(fitting_file)
        data_used_in_environment = cwd + '/data_exports/{}'.format(testing_file)
        self.sim.scaler.fit(self.sim.import_csv(filename=fitting_data_filepath))
        self.data = self.sim.import_csv(filename=data_used_in_environment)
        self.data = self.data.values
        self.observation = self.reset()

        self.action_space = spaces.Discrete(len(self.actions))
        variable_features_count = len(self.inventory_features) + len(self.actions)
        self.observation_space = spaces.Box(low=-np.inf,
                                            high=np.inf,
                                            shape=(1, len(self.features) + variable_features_count),
                                            dtype=np.float32)

    def step(self, action):
        if self.done:
            self.reset()
            self.done = False
            return self.observation, self.reward, self.done, {}

        position_features = self._create_position_features()
        action_features = self._create_action_features(action=action)

        self._midpoint = self.data[self.local_step_number][0]
        self.broker.step(midpoint=self._midpoint)
        self.reward = self._send_to_broker_and_get_reward(action)

        self.observation = np.concatenate((self.process_data(self.data[self.local_step_number]),
                                           position_features,
                                           action_features,
                                           np.array([self.reward])))

        self.data_for_render.append(np.concatenate((np.array([self._midpoint]), position_features, action_features)))

        # self.local_step_number += np.random.randint(low=1, high=10)
        self.local_step_number += self.step_size

        if self.local_step_number > self.data.shape[0] - 2:
            self.done = True
            order = Order(ccy=self.sym, side=None, price=self._midpoint, step=self.local_step_number)
            self.reward = self.broker.flatten_inventory(order=order)
            self.logger.info('{} is DONE.'.format(self.env_id))

        return self.observation, self.reward, self.done, {}

    def reset(self):
        self.logger.info(' {} reset. Episode pnl: {}'.format(self.env_id, self.broker.get_total_pnl(self._midpoint)))
        self.reward = 0.0
        self.done = False
        self.broker.reset()
        self.data_for_render.clear()

        if self.training:
            self.local_step_number = np.random.randint(low=0, high=self.data.shape[0] - 10)
        else:
            self.local_step_number = 0

        self.logger.info('  First step is %i' % self.local_step_number)

        position_features = self._create_position_features()
        action_features = self._create_action_features(action=0)

        self.observation = np.concatenate((self.process_data(self.data[self.local_step_number]),
                                           position_features,
                                           action_features,
                                           np.array([self.reward])))

        self._midpoint = self.data[self.local_step_number][0]
        self.data_for_render.append(np.concatenate((np.array([self._midpoint]), position_features, action_features)))

        self.local_step_number += self.step_size
        return self.observation

    def render(self, mode='human'):
        columns = ['midpoint'] + self.inventory_features + ['buy', 'sell']
        self.data_for_render = pd.DataFrame(self.data_for_render, columns=columns)
        print('attempting to plot data...')

        plt.figure(figsize=(20, 10))
        plt.title(self.sym)
        self.data_for_render['midpoint'].plot()

        trades = self.data_for_render.loc[self.data_for_render['buy'] == 1]
        for idx, trade in enumerate(trades.iterrows()):
            plt.plot(trade[0], trade[1][0], marker='^', color='lawngreen')

        trades = self.data_for_render.loc[self.data_for_render['sell'] == 1]
        for idx, trade in enumerate(trades.iterrows()):
            plt.plot(trade[0], trade[1][0], marker='v', color='red')

        plt.legend()
        plt.show()
        if False:
            plt.savefig('%s plot' % self.sym)

        self.data_for_render = []

    def close(self):
        self.logger.info('{} is being closed.'.format(self.env_id))
        self.data = None
        self.broker = None
        self.sim = None
        return

    def seed(self, seed=None):
        np_random, seed = seeding.np_random(seed)
        return [seed]

    def process_data(self, _next_state):
        return self.sim.scale_state(_next_state)

    def _send_to_broker_and_get_reward(self, action):
        reward = 0.0

        if action == 0:  # do nothing
            reward += 0.00001
            pass

        elif action == 1:  # buy
            price_fee_adjusted = self._midpoint + (self.fee * self._midpoint)
            if self.broker.short_inventory_count > 0:
                order = Order(ccy=self.sym, side='short', price=price_fee_adjusted, step=self.local_step_number)
                self.broker.remove(order=order)
                reward = self.broker.get_reward(side=order.side)

            elif self.broker.long_inventory_count >= 0:
                order = Order(ccy=self.sym, side='long', price=self._midpoint + self.fee, step=self.local_step_number)
                if self.broker.add(order=order) is False:
                    reward -= 0.0001

            else:
                self.logger.info('trading_gym.get_reward() Error for action #{} - unable to place an order with broker'
                                 .format(action))

        elif action == 2:  # sell
            price_fee_adjusted = self._midpoint - (self.fee * self._midpoint)
            if self.broker.long_inventory_count > 0:
                order = Order(ccy=self.sym, side='long', price=price_fee_adjusted, step=self.local_step_number)
                self.broker.remove(order=order)
                reward = self.broker.get_reward(side=order.side)
            elif self.broker.short_inventory_count >= 0:
                order = Order(ccy=self.sym, side='short', price=price_fee_adjusted, step=self.local_step_number)
                if self.broker.add(order=order) is False:
                    reward -= 0.00001

            else:
                self.logger.info('trading_gym.get_reward() Error for action #{} - unable to place an order with broker'
                                 .format(action))

        else:
            self.logger.info('Unknown action to take in get_reward(): action={} | midpoint={}'
                             .format(action, self._midpoint))

        return reward

    def _create_position_features(self):
        return np.array((self.broker.long_inventory.position_count / self.max_position,
                         self.broker.short_inventory.position_count / self.max_position,
                         self.broker.long_inventory.get_unrealized_pnl(self._midpoint),
                         self.broker.short_inventory.get_unrealized_pnl(self._midpoint)))

    def _create_action_features(self, action):
        return np.array(self.actions[action][1:])
