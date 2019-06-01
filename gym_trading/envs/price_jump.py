from gym import Env, spaces
from data_recorder.database.simulator import Simulator as Sim
from gym_trading.broker import Broker, Order
from gym_trading.render_env import TradingGraph
from configurations.configs import BROKER_FEE
import logging
import numpy as np


# logging
logging.basicConfig(level=logging.INFO, format='[%(asctime)s] %(message)s')
logger = logging.getLogger('PriceJump')


class PriceJump(Env):

    metadata = {'render.modes': ['human']}
    id = 'long-short-v0'
    action_repeats = 4
    inventory_features = ['long_inventory', 'short_inventory',
                          'total_unrealized_and_realized_pnl',
                          'long_unrealized_pnl', 'short_unrealized_pnl']
    # Turn to true if Bitifinex is in the dataset (e.g., include_bitfinex=True)
    features = Sim.get_feature_labels(include_system_time=False,
                                      include_bitfinex=False)
    instance = 0

    def __init__(self, training=True,
                 fitting_file='ETH-USD_2018-12-31.xz',
                 testing_file='ETH-USD_2019-01-01.xz',
                 step_size=1,
                 max_position=1,
                 window_size=50,
                 frame_stack=False):

        # properties required for instantiation
        PriceJump.instance += 1
        self._random_state = np.random.RandomState(seed=int(PriceJump.instance))
        self.seed = int(PriceJump.instance)
        self.training = training
        self.step_size = step_size
        self.fee = BROKER_FEE
        self.max_position = max_position
        self.window_size = window_size
        self.frame_stack = frame_stack
        self.frames_to_add = 3 if self.frame_stack else 0

        self.action = 0
        # derive gym.env properties
        self.actions = np.eye(3)

        self.sym = testing_file[:7]  # slice the CCY from the filename

        # properties that get reset()
        self.reward = 0.0
        self.done = False
        self.local_step_number = 0
        self.midpoint = 0.0
        self.observation = None

        # get Broker class to keep track of PnL and orders
        self.broker = Broker(max_position=max_position)
        # get historical data for simulations
        self.sim = Sim(use_arctic=False)

        fitting_data_filepath = '{}/data_exports/{}'.format(self.sim.cwd,
                                                            fitting_file)
        data_used_in_environment = '{}/data_exports/{}'.format(self.sim.cwd,
                                                               testing_file)
        # print('Fitting data: {}\nTesting Data: {}'.format(fitting_data_filepath,
    #                                                   data_used_in_environment))

        self.sim.fit_scaler(self.sim.import_csv(filename=fitting_data_filepath))
        self.data = self.sim.import_csv(filename=data_used_in_environment)
        self.prices_ = self.data['coinbase_midpoint'].values  # used to calculate PnL

        self.data_ = self.data.copy()
        logger.info("Pre-scaling {} data...".format(self.sym))
        self.data_ = self.data_.apply(self.sim.z_score, axis=1).values
        logger.info("...{} pre-scaling complete.".format(self.sym))
        self.data = self.data.values

        # rendering class
        self._render = TradingGraph(sym=self.sym)
        # graph midpoint prices
        self._render.reset_render_data(
            y_vec=self.prices_[:np.shape(self._render.x_vec)[0]])

        self.data_buffer, self.frame_stacker = list(), list()

        self.action_space = spaces.Discrete(len(self.actions))
        variable_features_count = len(self.inventory_features)+len(self.actions)+1
        if self.frame_stack is False:
            shape = (len(PriceJump.features) + variable_features_count,
                     self.window_size)
        else:
            shape = (len(PriceJump.features) + variable_features_count,
                     self.window_size, 4)

        self.observation_space = spaces.Box(low=self.data.min(),
                                            high=self.data.max(),
                                            shape=shape,
                                            dtype=np.int)

        print('PriceJump instantiated. ' +
              '\nself.observation_space.shape : {}'.format(
                  self.observation_space.shape))

    def __str__(self):
        return '{} | {}-{}'.format(PriceJump.id, self.sym, self.seed)

    def step(self, action):

        for current_step in range(PriceJump.action_repeats):

            if self.done:
                self.reset()
                return self.observation, self.reward, self.done

            # reset the reward if there ARE action repeats
            if current_step == 0:
                self.reward = 0.
                step_action = action
            else:
                step_action = 0

            self.midpoint = self.prices_[self.local_step_number]
            self.broker.step(midpoint=self.midpoint)
            self.reward += self._send_to_broker_and_get_reward(action=step_action)

            step_position_features = self._create_position_features()
            step_action_features = self._create_action_features(action=step_action)

            step_observation = np.concatenate((
                self.process_data(self.data_[self.local_step_number]),
                step_position_features,
                step_action_features,
                np.array([self.reward])),
                axis=None)
            self.data_buffer.append(step_observation)

            if len(self.data_buffer) >= self.window_size:
                self.frame_stacker.append(np.array(self.data_buffer,
                                                   dtype=np.float64))
                del self.data_buffer[0]

                if len(self.frame_stacker) > self.frames_to_add + 1:
                    del self.frame_stacker[0]

            self.local_step_number += self.step_size

        # output shape is [n_features, window_size, frames_to_add] eg [40, 100, 1]
        self.observation = np.array(self.frame_stacker, dtype=np.float64).transpose()

        # This removes a dimension to be compatible with the Keras-rl module
        # because Keras-rl uses its own frame-stacker. There are future plans
        # to integrate this repository with more reinforcement learning packages,
        # such as baselines.
        if self.frame_stack is False:
            self.observation = self.observation.reshape(
                self.observation.shape[0], -1)

        if self.local_step_number > self.data.shape[0] - 8:
            self.done = True
            order = Order(ccy=self.sym, side=None, price=self.midpoint,
                          step=self.local_step_number)
            self.reward = self.broker.flatten_inventory(order=order)

        return self.observation, self.reward, self.done, {}

    def reset(self):
        if self.training:
            self.local_step_number = self._random_state.randint(
                low=1,
                high=self.data.shape[0] // 5)
        else:
            self.local_step_number = 0

        logger.info(' %s-%i reset. Episode pnl: %.4f | First step: %i, max_pos: %i'
                    % (self.sym, self.seed,
                       self.broker.get_total_pnl(midpoint=self.midpoint),
                       self.local_step_number, self.max_position))
        self.reward = 0.0
        self.done = False
        self.broker.reset()
        self.data_buffer.clear()
        self.frame_stacker.clear()

        for step in range(self.window_size + self.frames_to_add):
            step_position_features = self._create_position_features()
            step_action_features = self._create_action_features(action=0)

            step_observation = np.concatenate((
                self.process_data(self.data_[self.local_step_number]),
                step_position_features,
                step_action_features,
                np.array([self.reward])),
                axis=None)
            self.data_buffer.append(step_observation)
            self.local_step_number += self.step_size

            if step >= self.window_size - 1:
                self.frame_stacker.append(
                    np.array(self.data_buffer, dtype=np.float64))
                del self.data_buffer[0]

                if len(self.frame_stacker) > self.frames_to_add + 1:
                    del self.frame_stacker[0]

        # output shape is [n_features, window_size, frames_to_add] eg [40, 100, 1]
        self.observation = np.array(self.frame_stacker, dtype=np.float64).transpose()

        # This removes a dimension to be compatible with the Keras-rl module
        # because Keras-rl uses its own frame-stacker. There are future plans
        # to integrate this repository with more reinforcement learning
        # packages, such as baselines.
        if self.frame_stack is False:
            self.observation = self.observation.reshape(
                self.observation.shape[0], -1)

        return self.observation

    def render(self, mode='human'):
        self._render.render(midpoint=self.midpoint, mode=mode)

    def close(self):
        logger.info('{}-{} is being closed.'.format(self.id, self.sym))
        self.data = None
        self.data_ = None
        self.prices_ = None
        self.broker = None
        self.sim = None
        self.data_buffer = None
        return

    def seed(self, seed=1):
        self._random_state = np.random.RandomState(seed=seed)
        self.seed = seed
        return [seed]

    @staticmethod
    def process_data(_next_state):
        return np.clip(_next_state.reshape((1, -1)), -10., 10.)

    # def process_data(self, _next_state):
    #     # return self.sim.scale_state(_next_state).values.reshape((1, -1))
    #     return np.reshape(_next_state, (1, -1))

    def _send_to_broker_and_get_reward(self, action):
        reward = 0.0

        if action == 0:  # do nothing
            pass

        elif action == 1:  # buy
            price_fee_adjusted = self.midpoint + (self.fee * self.midpoint)
            if self.broker.short_inventory_count > 0:
                order = Order(ccy=self.sym, side='short',
                              price=price_fee_adjusted,
                              step=self.local_step_number)
                self.broker.remove(order=order)
                reward += self.broker.get_reward(side=order.side)

            elif self.broker.long_inventory_count >= 0:
                order = Order(ccy=self.sym, side='long',
                              price=price_fee_adjusted,
                              step=self.local_step_number)
                if self.broker.add(order=order) is False:
                    reward -= 0.00000001

            else:
                logger.info(('gym_trading.get_reward() ' +
                             'Error for action #{} - ' +
                             'unable to place an order with broker').format(action))

        elif action == 2:  # sell
            price_fee_adjusted = self.midpoint - (self.fee * self.midpoint)
            if self.broker.long_inventory_count > 0:
                order = Order(ccy=self.sym, side='long',
                              price=price_fee_adjusted,
                              step=self.local_step_number)
                self.broker.remove(order=order)
                reward += self.broker.get_reward(side=order.side)
            elif self.broker.short_inventory_count >= 0:
                order = Order(ccy=self.sym, side='short',
                              price=price_fee_adjusted,
                              step=self.local_step_number)
                if self.broker.add(order=order) is False:
                    reward -= 0.00000001

            else:
                logger.info('gym_trading.get_reward() ' +
                            'Error for action #{} - ' +
                            'unable to place an order with broker'
                            .format(action))

        else:
            logger.info(('Unknown action to take in get_reward(): ' +
                         'action={} | midpoint={}').format(action, self.midpoint))

        return reward

    def _create_position_features(self):
        return np.array(
            (self.broker.long_inventory.position_count / self.max_position,
             self.broker.short_inventory.position_count / self.max_position,
             self.broker.get_total_pnl(midpoint=self.midpoint),
             self.broker.long_inventory.get_unrealized_pnl(self.midpoint),
             self.broker.short_inventory.get_unrealized_pnl(self.midpoint))
        )

    def _create_action_features(self, action):
        return self.actions[action]
