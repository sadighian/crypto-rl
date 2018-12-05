from gym import Env, spaces
from gym.utils import seeding
from simulator import Simulator as Sim
from broker import Broker
import logging
import numpy as np
from sklearn.preprocessing import MinMaxScaler


class TradingGym(Env):

    def __init__(self, data,
                 scaler,
                 training=True,
                 env_id='coinbase-bitfinex-v0',
                 step_size=1,
                 fee=0.006,
                 max_position=1):

        # properties required for instantiation
        self.training = training
        self.env_id = env_id
        self.step_size = step_size
        self.fee = fee
        self.max_position = max_position
        self.inventory_features = ['long_inventory', 'short_inventory']
        self.scaler = scaler

        # properties that get reset()
        self._reward = None
        self._done = False
        self._local_step_number = 0
        self._state = None
        self._next_state = None
        self._action = 0
        # derive gym.env properties
        self._actions = {
            0: (1, 0, 0),  # 0. do nothing
            1: (0, 1, 0),  # 1. buy
            2: (0, 0, 1)   # 2. sell
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
        observations_concatenated = len(self.features) + len(self.inventory_features) + len(self._actions)
        self.observation_space = spaces.Box(low=-np.inf,
                                            high=np.inf,
                                            shape=(1, observations_concatenated),
                                            dtype=np.float32)

        # logging
        logging.basicConfig(level=logging.INFO, format='[%(asctime)s] %(message)s')
        self.logger = logging.getLogger(env_id)
        self.logger.info('Making new env: {}'.format(env_id))

    def step(self, action):
        if self._done:
            self.logger.info('{} is done.'.format(self.env_id))
            self.observation = self.reset()
            return self.observation[1], self._reward, self._done, {}

        self._next_state = np.concatenate((self.process_data(self.data[self._local_step_number]),
                                           self.create_position_features(),
                                           self.create_action_features(action=self._action)))
        self.observation = (self._state, self._next_state)
        self._state = self._next_state

        self._midpoint = self.data[self._local_step_number][0]
        self._reward = self.send_to_broker_and_get_reward(action)

        self._action = action
        self._local_step_number += self.step_size
        if self._local_step_number > self.data.shape[0] - 2:
            self._done = True

        return self.observation[1], self._reward, self._done, {}

    def reset(self):
        print('{} has reset'.format(self.env_id))
        self._reward = None
        self._done = False
        self.broker.reset()
        self._local_step_number = 0
        self._state = np.concatenate((self.process_data(self.data[self._local_step_number]),
                                      self.create_position_features(),
                                      self.create_action_features(0)))
        self._local_step_number += 1
        self._next_state = np.concatenate((self.process_data(self.data[self._local_step_number]),
                                           self.create_position_features(),
                                           self.create_action_features(0)))
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

    def send_to_broker_and_get_reward(self, action):
        _reward = 0.0

        if action == 0:  # do nothing
            pass

        elif action == 1:  # buy
            if self.broker.short_inventory_count > 0:
                order = {
                    'price': self._midpoint,
                    'side': 'short'
                }
                self.broker.remove(order=order)
                _reward = self._get_reward(side=order['side'])

            elif self.broker.long_inventory_count >= 0:
                order = {
                    'price': self._midpoint,
                    'side': 'long'
                }
                self.broker.add(order=order)

            else:
                print('trading_gym.get_reward() Error for action #{} - unable to place an order with broker'
                      .format(action))

        elif action == 2:  # sell
            if self.broker.long_inventory_count > 0:
                order = {
                    'price': self._midpoint,
                    'side': 'long'
                }
                self.broker.remove(order=order)
                _reward = self._get_reward(side=order['side'])
            elif self.broker.short_inventory_count >= 0:
                order = {
                    'price': self._midpoint,
                    'side': 'short'
                }
                self.broker.add(order=order)
            else:
                print('trading_gym.get_reward() Error for action #{} - unable to place an order with broker'
                      .format(action))

        else:
            print('Unknown action to take in get_reward(): action={} | midpoint={}'.format(action, self._midpoint))

        return _reward

    def _get_reward(self, side='long'):
        if side == 'long':
            pnl_from_trade = self.broker.long_inventory.realized_pnl[-1] - self.fee
        elif side == 'short':
            pnl_from_trade = self.broker.short_inventory.realized_pnl[-1] - self.fee
        else:
            pnl_from_trade = -1.0
        pnl_multiple = pnl_from_trade / self.fee
        if pnl_multiple < 0.0:
            reward = -1.0
        elif pnl_multiple < 1.0:
            reward = 0.0
        elif pnl_multiple < 2.0:
            reward = 0.5
        else:
            reward = 1.0

        print('got reward: {}'.format(reward))
        return reward

    def create_action_features(self, action):
        return np.array(self._actions[action])

    def create_position_features(self):
        return np.array((self.broker.long_inventory.position_count / self.max_position,
                         self.broker.short_inventory.position_count / self.max_position))

    def process_data(self, _next_state):
        return self.scaler.transform(_next_state.reshape(1, -1)).reshape(_next_state.shape)
