from gym import Env, spaces
from data_recorder.database.simulator import Simulator as Sim
from gym_trading.broker2 import Broker, Order
from gym_trading.render_env import TradingGraph
from configurations.configs import BROKER_FEE
import logging
import numpy as np


# logging
logging.basicConfig(level=logging.INFO, format='[%(asctime)s] %(message)s')
logger = logging.getLogger('MarketMaker')


class MarketMaker(Env):
    # gym.env required
    metadata = {'render.modes': ['human']}
    id = 'market-maker-v0'
    # constants
    action_repeats = 4
    inventory_features = ['long_inventory', 'short_inventory',
                          'total_unrealized_and_realized_pnl',
                          'long_unrealized_pnl', 'short_unrealized_pnl',
                          'buy_distance_to_midpoint', 'short_distance_to_midpoint']
    # Turn to true if Bitifinex is in the dataset (e.g., include_bitfinex=True)
    features = Sim.get_feature_labels(include_system_time=False,
                                      include_bitfinex=False)
    best_bid_index = features.index('coinbase-bid-distance-0')
    best_ask_index = features.index('coinbase-ask-distance-0')
    notional_bid_index = features.index('coinbase-bid-notional-0')
    notional_ask_index = features.index('coinbase-ask-notional-0')

    buy_trade_index = features.index('coinbase-buys')
    sell_trade_index = features.index('coinbase-sells')

    target_pnl = BROKER_FEE * 10

    def __init__(self, training=True,
                 fitting_file='ETH-USD_2018-12-31.xz',
                 testing_file='ETH-USD_2019-01-01.xz',
                 step_size=1,
                 max_position=5,
                 window_size=40,
                 seed=1,
                 frame_stack=False):

        # properties required for instantiation
        self._random_state = np.random.RandomState(seed=seed)
        self._seed = seed
        self.training = training
        self.step_size = step_size
        self.fee = BROKER_FEE
        self.max_position = max_position
        self.window_size = window_size
        self.frame_stack = frame_stack
        self.frames_to_add = 3 if self.frame_stack else 0

        self._action = 0
        # derive gym.env properties
        self.actions = np.eye(17)

        self.sym = testing_file[:7]  # slice the CCY from the filename

        # properties that get reset()
        self.reward = 0.0
        self.done = False
        self._local_step_number = 0
        self.midpoint = 0.0
        self.observation = None

        # get Broker class to keep track of PnL and orders
        self.broker = Broker(max_position=max_position)
        # get historical data for simulations
        self.sim = Sim(use_arctic=False)

        fitting_data_filepath = '{}/data_exports/{}'.format(self.sim.cwd, fitting_file)
        data_used_in_environment = '{}/data_exports/{}'.format(self.sim.cwd, testing_file)
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
        self._render.reset_render_data(y_vec=self.prices_[:np.shape(self._render.x_vec)[0]])

        self.data_buffer, self.frame_stacker = list(), list()

        self.action_space = spaces.Discrete(len(self.actions))
        variable_features_count = len(self.inventory_features) + len(self.actions) + 1
        if self.frame_stack is False:
            shape = (len(MarketMaker.features) + variable_features_count, self.window_size)
        else:
            shape = (len(MarketMaker.features) + variable_features_count, self.window_size, 4)

        self.observation_space = spaces.Box(low=self.data.min(),
                                            high=self.data.max(),
                                            shape=shape,
                                            dtype=np.int)

        print('MarketMaker instantiated. ' +
              '\nself.observation_space.shape : {}'.format(
                  self.observation_space.shape))

    def __str__(self):
        return '{} | {}-{}'.format(MarketMaker.id, self.sym, self.seed)

    @property
    def step_number(self):
        return self._local_step_number

    def step(self, action):

        for current_step in range(MarketMaker.action_repeats):

            if self.done:
                self.reset()
                return self.observation, self.reward, self.done

            # reset the reward if there ARE action repeats
            if current_step == 0:
                self.reward = 0.
                step_action = action
            else:
                step_action = 0

            # Get current step's midpoint
            self.midpoint = self.prices_[self._local_step_number]
            # Pass current time step midpoint to broker to calculate PnL,
            # or if any open orders are to be filled
            step_best_bid, step_best_ask = self._get_nbbo()
            step_reward = self.broker.step(
                bid_price=step_best_bid,
                ask_price=step_best_ask,
                buy_volume=self._get_book_data(MarketMaker.buy_trade_index),
                sell_volume=self._get_book_data(MarketMaker.sell_trade_index),
                step=self._local_step_number
            ) / self.broker.reward_scale

            self.reward += self._send_to_broker_and_get_reward(step_action) + step_reward

            step_position_features = self._create_position_features()
            step_action_features = self._create_action_features(action=step_action)

            step_observation = np.concatenate((self.process_data(self.data_[self._local_step_number]),
                                               step_position_features,
                                               step_action_features,
                                               np.array([self.reward], dtype=np.float64)),
                                              axis=None)
            self.data_buffer.append(step_observation)

            if len(self.data_buffer) >= self.window_size:
                self.frame_stacker.append(np.array(self.data_buffer, dtype=np.float64))
                del self.data_buffer[0]

                if len(self.frame_stacker) > self.frames_to_add + 1:
                    del self.frame_stacker[0]

            self._local_step_number += self.step_size

        # output shape is [n_features, window_size, frames_to_add] e.g., [40, 100, 1]
        self.observation = np.array(self.frame_stacker, dtype=np.float64).transpose()

        # This removes a dimension to be compatible with the Keras-rl module
        # because Keras-rl uses its own frame-stacker. There are future plans to integrate
        # this repository with more reinforcement learning packages, such as baselines.
        if self.frame_stack is False:
            self.observation = self.observation.reshape(self.observation.shape[0], -1)

        if self._local_step_number > self.data.shape[0] - 8:
            self.done = True
            self.reward += self.broker.flatten_inventory(*self._get_nbbo())

        return self.observation, self.reward, self.done, {}

    def reset(self):
        if self.training:
            self._local_step_number = self._random_state.randint(low=1, high=self.data.shape[0]//5)
        else:
            self._local_step_number = 0

        logger.info(' {}-{} reset. Episode pnl: {} | First step: {}, max_pos: {}'.format(
            self.sym, self._seed,
            self.broker.get_total_pnl(midpoint=self.midpoint),
            self._local_step_number, self.max_position))
        self.reward = 0.0
        self.done = False
        self.broker.reset()
        self.data_buffer.clear()
        self.frame_stacker.clear()

        for step in range(self.window_size + self.frames_to_add):

            step_position_features = self._create_position_features()
            step_action_features = self._create_action_features(action=0)

            step_observation = np.concatenate((self.process_data(self.data_[self._local_step_number]),
                                               step_position_features,
                                               step_action_features,
                                               np.array([self.reward])),
                                              axis=None)
            self.data_buffer.append(step_observation)
            self._local_step_number += self.step_size

            if step >= self.window_size - 1:
                self.frame_stacker.append(np.array(self.data_buffer, dtype=np.float64))
                del self.data_buffer[0]

                if len(self.frame_stacker) > self.frames_to_add + 1:
                    del self.frame_stacker[0]

        # output shape is [n_features, window_size, frames_to_add] eg [40, 100, 1]
        self.observation = np.array(self.frame_stacker, dtype=np.float64).transpose()

        # This removes a dimension to be compatible with the Keras-rl module
        # because Keras-rl uses its own frame-stacker. There are future plans to integrate
        # this repository with more reinforcement learning packages, such as baselines.
        if self.frame_stack is False:
            self.observation = self.observation.reshape(self.observation.shape[0], -1)

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
        self._seed = seed
        return [seed]

    @staticmethod
    def process_data(_next_state):
        return np.clip(_next_state.reshape((1, -1)), -10., 10.)

    # def process_data(self, _next_state):
    #     # return self.sim.scale_state(_next_state).values.reshape((1, -1))
    #     return np.reshape(_next_state, (1, -1))

    def _send_to_broker_and_get_reward(self, action):
        reward = 0.0
        discouragement = 0.000000000001

        if action == 0:  # do nothing
            reward += discouragement

        elif action == 1:
            reward += self._create_order_at_level(reward, discouragement,
                                                  level=0, side='long')
            reward += self._create_order_at_level(reward, discouragement,
                                                  level=4, side='short')

        elif action == 2:
            reward += self._create_order_at_level(reward, discouragement,
                                                  level=0, side='long')
            reward += self._create_order_at_level(reward, discouragement,
                                                  level=9, side='short')
        elif action == 3:
            reward += self._create_order_at_level(reward, discouragement,
                                                  level=0, side='long')
            reward += self._create_order_at_level(reward, discouragement,
                                                  level=14, side='short')

        elif action == 4:
            reward += self._create_order_at_level(reward, discouragement,
                                                  level=4, side='long')
            reward += self._create_order_at_level(reward, discouragement,
                                                  level=0, side='short')

        elif action == 5:
            reward += self._create_order_at_level(reward, discouragement,
                                                  level=4, side='long')
            reward += self._create_order_at_level(reward, discouragement,
                                                  level=4, side='short')

        elif action == 6:

            reward += self._create_order_at_level(reward, discouragement,
                                                  level=4, side='long')
            reward += self._create_order_at_level(reward, discouragement,
                                                  level=9, side='short')
        elif action == 7:

            reward += self._create_order_at_level(reward, discouragement,
                                                  level=4, side='long')
            reward += self._create_order_at_level(reward, discouragement,
                                                  level=14, side='short')

        elif action == 8:
            reward += self._create_order_at_level(reward, discouragement,
                                                  level=9, side='long')
            reward += self._create_order_at_level(reward, discouragement,
                                                  level=0, side='short')

        elif action == 9:
            reward += self._create_order_at_level(reward, discouragement,
                                                  level=9, side='long')
            reward += self._create_order_at_level(reward, discouragement,
                                                  level=4, side='short')

        elif action == 10:
            reward += self._create_order_at_level(reward, discouragement,
                                                  level=9, side='long')
            reward += self._create_order_at_level(reward, discouragement,
                                                  level=9, side='short')

        elif action == 11:
            reward += self._create_order_at_level(reward, discouragement,
                                                  level=9, side='long')
            reward += self._create_order_at_level(reward, discouragement,
                                                  level=14, side='short')

        elif action == 12:
            reward += self._create_order_at_level(reward, discouragement,
                                                  level=14, side='long')
            reward += self._create_order_at_level(reward, discouragement,
                                                  level=0, side='short')

        elif action == 13:
            reward += self._create_order_at_level(reward, discouragement,
                                                  level=14, side='long')
            reward += self._create_order_at_level(reward, discouragement,
                                                  level=4, side='short')

        elif action == 14:
            reward += self._create_order_at_level(reward, discouragement,
                                                  level=14, side='long')
            reward += self._create_order_at_level(reward, discouragement,
                                                  level=9, side='short')

        elif action == 15:
            reward += self._create_order_at_level(reward, discouragement,
                                                  level=14, side='long')
            reward += self._create_order_at_level(reward, discouragement,
                                                  level=14, side='short')
        elif action == 16:
            reward += self.broker.flatten_inventory(*self._get_nbbo())
        else:
            logger.info("L'action n'exist pas ! Il faut faire attention !")

        return reward

    def _create_position_features(self):
        return np.array((self.broker.long_inventory.position_count / self.max_position,
                         self.broker.short_inventory.position_count / self.max_position,
                         self.broker.get_total_pnl(midpoint=self.midpoint) / MarketMaker.target_pnl,
                         self.broker.long_inventory.get_unrealized_pnl(self.midpoint) / self.broker.reward_scale,
                         self.broker.short_inventory.get_unrealized_pnl(self.midpoint) / self.broker.reward_scale,
                         self.broker.get_long_order_distance_to_midpoint(midpoint=self.midpoint) /
                         self.broker.reward_scale,
                         self.broker.get_short_order_distance_to_midpoint(midpoint=self.midpoint) /
                         self.broker.reward_scale))

    def _create_action_features(self, action):
        return self.actions[action]

    def _create_order_at_level(self, reward, discouragement, level=0, side='long'):
        adjustment = 1 if level > 0 else 0

        if side == 'long':
            best_bid = self._get_book_data(MarketMaker.best_bid_index + level)
            above_best_bid = round(self._get_book_data(MarketMaker.best_bid_index + level - adjustment), 2)
            price_improvement_bid = round(best_bid + 0.01, 2)

            if above_best_bid == price_improvement_bid:
                bid_price = round(self.midpoint - best_bid, 2)
                bid_queue_ahead = self._get_book_data(MarketMaker.notional_bid_index)
            else:
                bid_price = round(self.midpoint - price_improvement_bid, 2)
                bid_queue_ahead = 0.

            bid_order = Order(ccy=self.sym, side='long', price=bid_price, step=self._local_step_number,
                              queue_ahead=bid_queue_ahead)

            if self.broker.add(order=bid_order) is False:
                reward -= discouragement
            else:
                reward += discouragement

        if side == 'short':
            best_ask = self._get_book_data(MarketMaker.best_bid_index + level)
            above_best_ask = round(self._get_book_data(MarketMaker.best_ask_index + level - adjustment), 2)
            price_improvement_ask = round(best_ask - 0.01, 2)

            if above_best_ask == price_improvement_ask:
                ask_price = round(self.midpoint + best_ask, 2)
                ask_queue_ahead = self._get_book_data(MarketMaker.notional_ask_index)
            else:
                ask_price = round(self.midpoint + price_improvement_ask, 2)
                ask_queue_ahead = 0.

            ask_order = Order(ccy=self.sym, side='short', price=ask_price, step=self._local_step_number,
                              queue_ahead=ask_queue_ahead)

            if self.broker.add(order=ask_order) is False:
                reward -= discouragement
            else:
                reward += discouragement

        return reward

    def _get_nbbo(self):
        best_bid = round(self.midpoint - self._get_book_data(MarketMaker.best_bid_index), 2)
        best_ask = round(self.midpoint + self._get_book_data(MarketMaker.best_ask_index), 2)
        return best_bid, best_ask

    def _get_book_data(self, index=0):
        return self.data[self._local_step_number][index]
