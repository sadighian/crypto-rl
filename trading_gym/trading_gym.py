from gym import Env, spaces
from gym.utils import seeding
from .simulator import Simulator as Sim
from .broker import Broker
import logging
import numpy as np
import os


class TradingGym(Env):

    def __init__(self,
                 training=True, env_id='coinbasepro-bitfinex-v0',
                 step_size=1, fee=0.003, max_position=1):

        # properties required for instantiation
        self.training = training
        self.env_id = env_id
        self.step_size = step_size
        self.fee = fee
        self.max_position = max_position
        self.inventory_features = ['long_inventory', 'short_inventory', 'long_unrealized_pnl', 'short_unrealized_pnl']

        # properties that get reset()
        self.reward = 0.0
        self.prev_total_pnl = 0.0
        self.done = False
        self.local_step_number = 0
        self._state = None
        self._midpoint = 0.0

        self._action = 0
        # derive gym.env properties
        self.actions = {
            0: (1, 0, 0),  # 0. do nothing
            1: (0, 1, 0),  # 1. buy
            2: (0, 0, 1)  # 2. sell
        }

        # get historical data for simulations
        self.broker = Broker(max_position=max_position)
        self.sim = Sim()
        self.features = self.sim.get_feature_labels(include_system_time=False, lags=0)

        # For when you want to write the environment data to the disk, rather than generate
        # it everytime a simulation is run.

        # cwd = os.getcwd()
        cwd = os.path.dirname(os.path.realpath(__file__))
        fitting_data_filepath = cwd + '/data_exports/LTC-USD_20181120.csv'
        data_used_in_environment = cwd + '/data_exports/LTC-USD_20181121.csv'
        self.data = self.sim.load_env_states(fitting_filepath=fitting_data_filepath,
                                             env_filepath=data_used_in_environment)

        # The code below creates the environment data through querying the
        # Arctic Tick Store, replaying tick data to reconstruct the limit
        # order book, and extract its stationary features. This entire process
        # generally takes between 4-8 minutes (depending on the date queried).

        # self.data = self.sim.query_env_states(query={
        #     'ccy': ['ETC-USD', 'tETCUSD'],
        #     'start_date': 20181121,
        #     'end_date': 20181122
        # })
        self.data = self.data.values
        self.observation = self.reset()

        self.action_space = spaces.Discrete(len(self.actions))
        self.observation_space = spaces.Box(low=-np.inf,
                                            high=np.inf,
                                            shape=(1, len(self.features) +
                                                   len(self.inventory_features) +
                                                   len(self.actions) - 1),
                                            dtype=np.float32)
        self.env_id = env_id

        # logging
        logging.basicConfig(level=logging.INFO, format='[%(asctime)s] %(message)s')
        self.logger = logging.getLogger(env_id)
        self.logger.info('Making new env: {}'.format(env_id))

    def step(self, action):
        if self.done:
            self.logger.info('***  {} is done.   ***'.format(self.env_id))
            self.observation = self.reset()
            self.done = False
            return self.observation, self.reward, self.done, {}

        _next_state = np.concatenate((self.process_data(self.data[self.local_step_number]),
                                      self.create_position_features(),
                                      self.create_action_features(action=action)))
        self.observation = (self._state, _next_state)
        self._state = _next_state

        self._midpoint = self.data[self.local_step_number][0]
        self.reward = self.send_to_broker_and_get_reward(action)

        # TODO undo random
        # self.local_step_number += np.random.randint(low=1, high=10)
        self.local_step_number += self.step_size
        if self.local_step_number > self.data.shape[0] - 2:
            self.done = True
            order = {
                'price': self._midpoint,
                'step': self.local_step_number
            }
            self.reward = self.broker.flatten_inventory(order=order)

        return self.observation[1], self.reward, self.done, {}

    def reset(self):
        print(' {} reset. Episode pnl: {}'.format(self.env_id, self.broker.get_total_pnl(self._midpoint)))
        self.reward = 0.0
        self.done = False
        self.broker.reset()

        if self.training:
            self.local_step_number = np.random.randint(low=0, high=self.data.shape[0] - 10)
        else:
            self.local_step_number = 0

        print('  First step is %i' % self.local_step_number)

        _prev_state = np.concatenate((self.process_data(self.data[self.local_step_number]),
                                      self.create_position_features(),
                                      self.create_action_features(0)))
        self.local_step_number += 1
        _next_state = np.concatenate((self.process_data(self.data[self.local_step_number]),
                                      self.create_position_features(),
                                      self.create_action_features(0)))
        self.local_step_number += self.step_size
        self._state = _prev_state
        _pair_state = (_prev_state, _next_state)
        return _pair_state[1]

    def render(self, mode='human'):
        pass

    def close(self):
        self.data = None
        self.broker = None
        self.sim = None
        return

    def seed(self, seed=None):
        np_random, seed = seeding.np_random(seed)
        return [seed]

    def send_to_broker_and_get_reward(self, action):
        _reward = 0.0

        if action == 0:  # do nothing
            pass

        elif action == 1:  # buy
            if self.broker.short_inventory_count > 0:
                order = {
                    'price': self._midpoint,
                    'side': 'short',
                    'step': self.local_step_number
                }
                self.broker.remove(order=order)
                _reward = self._get_reward(side=order['side'])

            elif self.broker.long_inventory_count >= 0:
                order = {
                    'price': self._midpoint,
                    'side': 'long',
                    'step': self.local_step_number
                }
                if self.broker.add(order=order) is False:
                    _reward -= self.fee
                # print('  Long @ step {}'.format(self.local_step_number))

            else:
                print('trading_gym.get_reward() Error for action #{} - unable to place an order with broker'
                      .format(action))

        elif action == 2:  # sell
            if self.broker.long_inventory_count > 0:
                order = {
                    'price': self._midpoint,
                    'side': 'long',
                    'step': self.local_step_number
                }
                self.broker.remove(order=order)
                _reward = self._get_reward(side=order['side'])
            elif self.broker.short_inventory_count >= 0:
                order = {
                    'price': self._midpoint,
                    'side': 'short',
                    'step': self.local_step_number
                }
                if self.broker.add(order=order) is False:
                    _reward -= self.fee
                # print('  Short @ step {}'.format(self.local_step_number))

            else:
                print('trading_gym.get_reward() Error for action #{} - unable to place an order with broker'
                      .format(action))

        else:
            print('Unknown action to take in get_reward(): action={} | midpoint={}'.format(action, self._midpoint))

        return _reward

    # def _get_reward(self, side='long'):
    #     if side == 'long':
    #         pnl_from_trade = self.broker.long_inventory.realized_pnl[-1] - self.fee
    #         steps_in_position = self.broker.long_inventory.steps_in_position[-1]
    #     elif side == 'short':
    #         pnl_from_trade = self.broker.short_inventory.realized_pnl[-1] - self.fee
    #         steps_in_position = self.broker.short_inventory.steps_in_position[-1]
    #     else:
    #         pnl_from_trade = -1.0
    #         steps_in_position = 0
    #         print('*trading_gym._get_reward: Unknown order side: {}'.format(side))
    #
    #     pnl_multiple = pnl_from_trade / self.fee
    #     if pnl_multiple < 0.0:
    #         reward = max(-1.0, pnl_multiple)
    #         # print('Bad reward: {}'.format(reward))
    #     elif pnl_multiple < 1.0:
    #         reward = 0.01
    #     elif pnl_multiple < 2.0:
    #         reward = 0.4
    #         print('  Ok reward: {} | {} steps'.format(reward, steps_in_position))
    #     elif pnl_multiple < 3.0:
    #         reward = 0.8
    #         print('  Good reward: {} | {} steps'.format(reward, steps_in_position))
    #     else:
    #         reward = 1.0
    #         print('  Fantastic reward: {} | {} steps'.format(reward, steps_in_position))
    #
    #     return reward

    def _get_reward(self, side='long'):
        if side == 'long':
            reward = self.broker.long_inventory.realized_pnl[-1] - self.fee
            steps_in_position = self.broker.long_inventory.steps_in_position[-1]
        elif side == 'short':
            reward = self.broker.short_inventory.realized_pnl[-1] - self.fee
            steps_in_position = self.broker.short_inventory.steps_in_position[-1]
        else:
            reward = -1.0
            steps_in_position = 0
            print('*trading_gym._get_reward: Unknown order side: {}'.format(side))

        reward_multiple = reward / self.fee
        if reward_multiple > 2.0:
            print('  %s REWARD: %.2f | %i of %i' % (side, reward_multiple, steps_in_position, self.local_step_number))

        return reward

    def create_position_features(self):
        return np.array((self.broker.long_inventory.position_count / self.max_position,
                         self.broker.short_inventory.position_count / self.max_position,
                         self.broker.long_inventory.get_unrealized_pnl(self._midpoint),
                         self.broker.short_inventory.get_unrealized_pnl(self._midpoint)))

    def process_data(self, _next_state):
        return self.sim.scale_state(_next_state)

    def create_action_features(self, action):
        return np.array(self.actions[action][1:])
