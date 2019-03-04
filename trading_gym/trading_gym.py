from gym import Env, spaces
from simulator import Simulator as Sim
from broker import Broker, Order
import logging
import numpy as np
import os


class TradingGym(Env):

    def __init__(self, training=True,
                 fitting_file='LTC-USD_20181120.xz',
                 testing_file='LTC-USD_20181121.xz',
                 env_id='coinbasepro-bitfinex-v0',
                 step_size=1,
                 fee=0.003,
                 max_position=1,
                 window_size=50,
                 seed=1):

        # properties required for instantiation
        self._random_state = np.random.RandomState(seed=seed)
        self._seed = seed
        self.training = training
        self.env_id = env_id
        self.step_size = step_size
        self.fee = fee
        self.max_position = max_position
        self.window_size = window_size
        self.inventory_features = ['long_inventory', 'short_inventory',
                                   'long_unrealized_pnl', 'short_unrealized_pnl']
        self._action = 0
        # derive gym.env properties
        self.actions = ((1, 0, 0),  # 0. do nothing
                        (0, 1, 0),  # 1. buy
                        (0, 0, 1)  # 2. sell
                        )
        self.sym = testing_file[:7]  # slice the CCY from the filename

        # logging
        logging.basicConfig(level=logging.INFO, format='[%(asctime)s] %(message)s')
        self.logger = logging.getLogger(env_id)

        # properties that get reset()
        self.reward = 0.0
        self.done = False
        self.local_step_number = 0
        self._state = None
        self._midpoint = 0.0
        self.observation = None
        self.data_buffer = []

        # get historical data for simulations
        self.broker = Broker(max_position=max_position)
        self.sim = Sim(use_arctic=False)

        # Turn to true if Bitifinex is in the dataset (e.g., include_bitfinex=True)
        self.features = self.sim.get_feature_labels(include_system_time=False, include_bitfinex=False)

        # cwd = os.getcwd()
        cwd = os.path.dirname(os.path.realpath(__file__))
        fitting_data_filepath = cwd + '/data_exports/{}'.format(fitting_file)
        data_used_in_environment = cwd + '/data_exports/{}'.format(testing_file)
        print('Fitting data: {}\nTesting Data: {}'.format(fitting_data_filepath, data_used_in_environment))

        self.sim.fit_scaler(self.sim.import_csv(filename=fitting_data_filepath))
        self.data = self.sim.import_csv(filename=data_used_in_environment)
        self.data = self.data.values

        self.action_space = spaces.Discrete(len(self.actions))
        variable_features_count = len(self.inventory_features) + len(self.actions) + 1
        self.observation_space = spaces.Box(low=-self.data.min(),
                                            high=self.data.max(),
                                            shape=(self.window_size, len(self.features) + variable_features_count),
                                            dtype=np.float32)
        self.reset()
        print('self.observation_space.shape : {}'.format(self.observation_space.shape))

    def __str__(self):
        return '{} | {}-{}'.format(self.env_id, self.sym, self.seed)

    def step(self, action, num_steps=4):

        for current_step in range(num_steps):

            if self.done:
                self.reset()
                self.done = False
                return self.observation, self.reward, self.done

            position_features = self._create_position_features()
            action_features = self._create_action_features(action=action)

            self._midpoint = self.data[self.local_step_number][0]
            self.broker.step(midpoint=self._midpoint)

            if current_step == 0:
                self.reward = self._send_to_broker_and_get_reward(action)
            else:
                self.reward += self._send_to_broker_and_get_reward(action)

            _observation = np.concatenate((self.process_data(self.data[self.local_step_number]),
                                           position_features,
                                           action_features,
                                           np.array([self.reward])),
                                          axis=None)
            self.data_buffer.insert(0, _observation)

            if len(self.data_buffer) >= self.window_size:
                self.data_buffer.pop()

            self.local_step_number += self.step_size

        self.observation = np.array(self.data_buffer,
                                    dtype=np.float32).reshape(self.observation_space.shape)

        if self.local_step_number > self.data.shape[0] - 8:
            self.done = True
            order = Order(ccy=self.sym, side=None, price=self._midpoint, step=self.local_step_number)
            self.reward = self.broker.flatten_inventory(order=order)

        return self.observation, self.reward, self.done

    def reset(self):
        if self.training:
            self.local_step_number = self._random_state.randint(low=1, high=5000)
        else:
            self.local_step_number = 0

        self.logger.info(' %s-%i reset | Episode pnl: %.4f | First step: %i | max_pos: %i'
                         % (self.sym, self._seed,
                            self.broker.get_total_pnl(midpoint=self._midpoint),
                            self.local_step_number, self.max_position))
        self.reward = 0.0
        self.done = False
        self.broker.reset()
        self.data_buffer.clear()

        for step in range(self.window_size):
            position_features = self._create_position_features()
            action_features = self._create_action_features(action=0)

            _observation = np.concatenate((self.process_data(self.data[self.local_step_number]),
                                           position_features,
                                           action_features,
                                           np.array([self.reward])),
                                          axis=None)
            self.data_buffer.insert(0, _observation)
            self.local_step_number += self.step_size

        self.observation = np.array(self.data_buffer,
                                    dtype=np.float32).reshape(self.observation_space.shape)
        print('{} reset.observation.shape = {}'.format(self.sym, np.shape(self.observation)))

        return self.observation

    def render(self, mode='human'):
        pass

    def close(self):
        self.logger.info('{} is being closed.'.format(self.env_id))
        self.data = None
        self.broker = None
        self.sim = None
        self.data_buffer = None
        return

    def seed(self, seed=1):
        self._random_state = np.random.RandomState(seed=seed)
        self._seed = seed
        return [seed]

    def process_data(self, _next_state):
        return self.sim.scale_state(_next_state).values.reshape((1, -1))

    def _send_to_broker_and_get_reward(self, action):
        reward = 0.0

        if action == 0:  # do nothing
            reward += 0.00000001
            pass

        elif action == 1:  # buy
            price_fee_adjusted = self._midpoint + (self.fee * self._midpoint)
            if self.broker.short_inventory_count > 0:
                order = Order(ccy=self.sym, side='short', price=price_fee_adjusted, step=self.local_step_number)
                self.broker.remove(order=order)
                reward += self.broker.get_reward(side=order.side)

            elif self.broker.long_inventory_count >= 0:
                order = Order(ccy=self.sym, side='long', price=self._midpoint + self.fee, step=self.local_step_number)
                if self.broker.add(order=order) is False:
                    reward -= 0.00000001

            else:
                self.logger.info('trading_gym.get_reward() Error for action #{} - unable to place an order with broker'
                                 .format(action))

        elif action == 2:  # sell
            price_fee_adjusted = self._midpoint - (self.fee * self._midpoint)
            if self.broker.long_inventory_count > 0:
                order = Order(ccy=self.sym, side='long', price=price_fee_adjusted, step=self.local_step_number)
                self.broker.remove(order=order)
                reward += self.broker.get_reward(side=order.side)
            elif self.broker.short_inventory_count >= 0:
                order = Order(ccy=self.sym, side='short', price=price_fee_adjusted, step=self.local_step_number)
                if self.broker.add(order=order) is False:
                    reward -= 0.00000001

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
        return np.array(self.actions[action])
