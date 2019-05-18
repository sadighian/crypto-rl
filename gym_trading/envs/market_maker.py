from gym import Env, spaces
from data_recorder.database.simulator import Simulator as Sim
from gym_trading.broker2 import Broker, Order
from configurations.configs import BROKER_FEE
import logging
import numpy as np
import matplotlib.pyplot as plt


plt.style.use('dark_background')


# logging
logging.basicConfig(level=logging.INFO, format='[%(asctime)s] %(message)s')
logger = logging.getLogger('MarketMaker')


class MarketMaker(Env):

    metadata = {'render.modes': ['human']}
    id = 'market-maker-v0'
    action_repeats = 4
    bid_price_features = ['coinbase-bid-distance-0', 'coinbase-bid-distance-1',
                          'coinbase-bid-distance-2', 'coinbase-bid-distance-3',
                          'coinbase-bid-distance-4', 'coinbase-bid-distance-5',
                          'coinbase-bid-distance-6', 'coinbase-bid-distance-7',
                          'coinbase-bid-distance-8', 'coinbase-bid-distance-9']
    ask_price_features = ['coinbase-ask-distance-0', 'coinbase-ask-distance-1',
                          'coinbase-ask-distance-2', 'coinbase-ask-distance-3',
                          'coinbase-ask-distance-4', 'coinbase-ask-distance-5',
                          'coinbase-ask-distance-6', 'coinbase-ask-distance-7',
                          'coinbase-ask-distance-8', 'coinbase-ask-distance-9']
    bid_notional_features = ['coinbase-bid-notional-0', 'coinbase-bid-notional-1',
                             'coinbase-bid-notional-2', 'coinbase-bid-notional-3',
                             'coinbase-bid-notional-4', 'coinbase-bid-notional-5',
                             'coinbase-bid-notional-6', 'coinbase-bid-notional-7',
                             'coinbase-bid-notional-8', 'coinbase-bid-notional-9']
    ask_notional_features = ['coinbase-ask-notional-0', 'coinbase-ask-notional-1',
                             'coinbase-ask-notional-2', 'coinbase-ask-notional-3',
                             'coinbase-ask-notional-4', 'coinbase-ask-notional-5',
                             'coinbase-ask-notional-6', 'coinbase-ask-notional-7',
                             'coinbase-ask-notional-8', 'coinbase-ask-notional-9']

    def __init__(self, training=True,
                 fitting_file='ETH-USD_2018-12-31.xz',
                 testing_file='ETH-USD_2019-01-01.xz',
                 step_size=1,
                 max_position=5,
                 window_size=50,
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
        self.inventory_features = ['long_inventory', 'short_inventory',
                                   'long_unrealized_pnl', 'short_unrealized_pnl',
                                   'buy_distance_to_midpoint', 'short_distance_to_midpoint']

        self._action = 0
        # derive gym.env properties
        self.actions = np.eye(24)

        self.sym = testing_file[:7]  # slice the CCY from the filename

        # properties that get reset()
        self.reward = 0.0
        self.done = False
        self._local_step_number = 0
        self.midpoint = 0.0
        self.observation = None

        # get historical data for simulations
        self.broker = Broker(max_position=max_position)
        self.sim = Sim(use_arctic=False)

        # Turn to true if Bitifinex is in the dataset (e.g., include_bitfinex=True)
        self.features = self.sim.get_feature_labels(include_system_time=False,
                                                    include_bitfinex=False)

        fitting_data_filepath = '{}/data_exports/{}'.format(self.sim.cwd, fitting_file)
        data_used_in_environment = '{}/data_exports/{}'.format(self.sim.cwd, testing_file)
        # print('Fitting data: {}\nTesting Data: {}'.format(fitting_data_filepath,
        #                                                   data_used_in_environment))

        self.sim.fit_scaler(self.sim.import_csv(filename=fitting_data_filepath))
        self.data = self.sim.import_csv(filename=data_used_in_environment)
        self.prices = self.data['coinbase_midpoint'].values
        self.bid_prices = self.data[MarketMaker.bid_price_features].values
        self.ask_prices = self.data[MarketMaker.ask_price_features].values
        self.bid_notionals = self.data[MarketMaker.bid_notional_features].values
        self.ask_notionals = self.data[MarketMaker.ask_notional_features].values

        # self.data = self.data.apply(self.sim.z_score, axis=1)
        self.data_ = self.data.copy()
        self.data = self.data.values
        # self.data = None

        self.data_buffer, self.frame_stacker = list(), list()
        self.action_space = spaces.Discrete(len(self.actions))
        variable_features_count = len(self.inventory_features) + len(self.actions) + 1

        if self.frame_stack is False:
            shape = (len(self.features) + variable_features_count, self.window_size)
        else:
            shape = (len(self.features) + variable_features_count, self.window_size, 4)

        self.observation_space = spaces.Box(low=self.data.min(),
                                            high=self.data.max(),
                                            shape=shape,
                                            dtype=np.int)

        # attributes for rendering
        self.line1 = []
        self.screen_size = 1000
        self.y_vec = None
        self.x_vec = None
        self._reset_render_data()

        self.reset()
        # print('MarketMaker instantiated. ' +
        #       '\nself.observation_space.shape : {}'.format(
        #           self.observation_space.shape))

    def __str__(self):
        return '{} | {}-{}'.format(MarketMaker.id, self.sym, self.seed)

    def _reset_render_data(self):
        self.x_vec = np.linspace(0, self.screen_size * 10, self.screen_size + 1)[0:-1]
        self.y_vec = np.array(self.prices[:np.shape(self.x_vec)[0]])
        self.line1 = []

    @property
    def step_number(self):
        return self._local_step_number

    def step(self, action_):

        for current_step in range(MarketMaker.action_repeats):

            if self.done:
                self.reset()
                return self.observation, self.reward, self.done

            # reset the reward if there are action repeats
            if current_step == 0:
                self.reward = 0.
                action = action_
            else:
                action = 0

            # Get current step's midpoint to calculate PnL, or if
            # an open order got filled.
            self.midpoint = self.prices[self._local_step_number]
            _step_reward = self.broker.step(
                bid_price=self.midpoint - self.bid_prices[self._local_step_number][0],
                ask_price=self.midpoint + self.ask_prices[self._local_step_number][0],
                buy_volume=self.data[self._local_step_number][-2],
                sell_volume=self.data[self._local_step_number][-1],
                step=self._local_step_number
            )

            self.reward += self._send_to_broker_and_get_reward(action) + _step_reward

            position_features = self._create_position_features()
            action_features = self._create_action_features(action=action)

            _observation = np.concatenate((self.process_data(self.data[self._local_step_number]),
                                           position_features,
                                           action_features,
                                           np.array([self.reward])),
                                          axis=None)
            self.data_buffer.append(_observation)

            if len(self.data_buffer) >= self.window_size:
                self.frame_stacker.append(np.array(self.data_buffer, dtype=np.float32))
                del self.data_buffer[0]

                if len(self.frame_stacker) > self.frames_to_add + 1:
                    del self.frame_stacker[0]

            self._local_step_number += self.step_size

        # output shape is [n_features, window_size, frames_to_add] e.g., [40, 100, 1]
        self.observation = np.array(self.frame_stacker, dtype=np.float32).transpose()

        # This removes a dimension to be compatible with the Keras-rl module
        # because Keras-rl uses its own frame-stacker. There are future plans to integrate
        # this repository with more reinforcement learning packages, such as baselines.
        if self.frame_stack is False:
            self.observation = self.observation.reshape(self.observation.shape[0], -1)

        if self._local_step_number > self.data.shape[0] - 8:
            self.done = True
            best_bid = round(self.midpoint + self.bid_prices[self._local_step_number][0], 2)
            best_ask = round(self.midpoint + self.ask_prices[self._local_step_number][0], 2)
            self.reward += self.broker.flatten_inventory(bid_price=best_bid, ask_price=best_ask)

        return self.observation, self.reward, self.done, {}

    def reset(self):
        if self.training:
            self._local_step_number = self._random_state.randint(low=1, high=5000)
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

        self._reset_render_data()

        for step in range(self.window_size + self.frames_to_add):

            position_features = self._create_position_features()
            action_features = self._create_action_features(action=0)

            _observation = np.concatenate((self.process_data(self.data[self._local_step_number]),
                                           position_features,
                                           action_features,
                                           np.array([self.reward])),
                                          axis=None)
            self.data_buffer.append(_observation)
            self._local_step_number += self.step_size

            if step >= self.window_size - 1:
                self.frame_stacker.append(np.array(self.data_buffer, dtype=np.float32))
                del self.data_buffer[0]

                if len(self.frame_stacker) > self.frames_to_add + 1:
                    del self.frame_stacker[0]

        # output shape is [n_features, window_size, frames_to_add] eg [40, 100, 1]
        self.observation = np.array(self.frame_stacker, dtype=np.float32).transpose()

        # This removes a dimension to be compatible with the Keras-rl module
        # because Keras-rl uses its own frame-stacker. There are future plans to integrate
        # this repository with more reinforcement learning packages, such as baselines.
        if self.frame_stack is False:
            self.observation = self.observation.reshape(self.observation.shape[0], -1)

        return self.observation

    def render(self, mode='human'):
        if mode == 'human':
            self.line1 = _live_plotter(self.x_vec,
                                       self.y_vec,
                                       self.line1,
                                       identifier=self.sym)
            self.y_vec = np.append(self.y_vec[1:], self.midpoint)

    def close(self):
        logger.info('{}-{} is being closed.'.format(self.id, self.sym))
        self.data = None
        self.broker = None
        self.sim = None
        self.data_buffer = None
        return

    def seed(self, seed=1):
        self._random_state = np.random.RandomState(seed=seed)
        self._seed = seed
        return [seed]

    # @staticmethod
    # def process_data(_next_state):
    #     return np.clip(_next_state.reshape((1, -1)), -10., 10.)

    def process_data(self, _next_state):
        # return self.sim.scale_state(_next_state).values.reshape((1, -1))
        return np.reshape(_next_state, (1, -1))

    def _send_to_broker_and_get_reward(self, action):
        reward = 0.0
        discouragement = 0.000000000001

        if action == 0:  # do nothing
            reward += discouragement

        elif action == 1:  # set bid to inside spread or [ask_price - 0.01]
            best_bid = self.bid_prices[self._local_step_number][0]
            best_ask = self.ask_prices[self._local_step_number][0]
            price = round(max(self.midpoint - best_bid, self.midpoint + best_ask - 0.01), 2)
            order = Order(ccy=self.sym, side='long', price=price, step=self._local_step_number)
            if self.broker.add(order=order) is False:
                reward -= discouragement

        elif action == 2:  # set bid to best_bid - row 0
            reward = self._create_bid_order_at_level(reward, discouragement, 0)

        elif action == 3:  # set bid to best_bid - row 1
            reward = self._create_bid_order_at_level(reward, discouragement, 1)

        elif action == 4:  # set bid to best_bid - row 2
            reward = self._create_bid_order_at_level(reward, discouragement, 2)

        elif action == 5:  # set bid to best_bid - row 3
            reward = self._create_bid_order_at_level(reward, discouragement, 3)

        if action == 6:  # set bid to best_bid - row 4
            reward = self._create_bid_order_at_level(reward, discouragement, 4)

        elif action == 7:  # set bid to best_bid - row 5
            reward = self._create_bid_order_at_level(reward, discouragement, 5)

        elif action == 8:  # set bid to best_bid - row 6
            reward = self._create_bid_order_at_level(reward, discouragement, 6)

        elif action == 9:  # set bid to best_bid - row 7
            reward = self._create_bid_order_at_level(reward, discouragement, 7)

        if action == 10:  # set bid to best_bid - row 8
            reward = self._create_bid_order_at_level(reward, discouragement, 8)

        elif action == 11:  # set bid to best_bid - row 9
            reward = self._create_bid_order_at_level(reward, discouragement, 9)

        elif action == 12:  # set ask to inside spread or [bid_price + 0.01]
            best_bid = self.bid_prices[self._local_step_number][0]
            best_ask = self.ask_prices[self._local_step_number][0]
            price = round(min(best_ask + self.midpoint, self.midpoint - best_bid + 0.01), 2)
            order = Order(ccy=self.sym, side='long', price=price, step=self._local_step_number)
            if self.broker.add(order=order) is False:
                reward -= discouragement

        if action == 13:  # set ask to best_bid - row 0
            reward = self._create_ask_order_at_level(reward, discouragement, 0)

        elif action == 14:  # set ask to best_bid - row 1
            reward = self._create_ask_order_at_level(reward, discouragement, 1)

        elif action == 15:  # set ask to best_bid - row 2
            reward = self._create_ask_order_at_level(reward, discouragement, 2)

        if action == 16:  # set ask to best_bid - row 3
            reward = self._create_ask_order_at_level(reward, discouragement, 3)

        elif action == 17:  # set ask to best_bid - row 4
            reward = self._create_ask_order_at_level(reward, discouragement, 4)

        elif action == 18:  # set ask to best_bid - row 5
            reward = self._create_ask_order_at_level(reward, discouragement, 5)

        if action == 19:  # set ask to best_bid - row 6
            reward = self._create_ask_order_at_level(reward, discouragement, 6)

        elif action == 20:  # set ask to best_bid - row 7
            reward = self._create_ask_order_at_level(reward, discouragement, 7)

        elif action == 21:  # set ask to best_bid - row 8
            reward = self._create_ask_order_at_level(reward, discouragement, 8)

        elif action == 22:  # set ask to best_bid - row 9
            reward = self._create_ask_order_at_level(reward, discouragement, 9)

        if action == 23:  # flatten all positions
            best_bid = round(self.midpoint + self.bid_prices[self._local_step_number][0], 2)
            best_ask = round(self.midpoint + self.ask_prices[self._local_step_number][0], 2)
            reward += self.broker.flatten_inventory(bid_price=best_bid, ask_price=best_ask)

        elif action == 24:  #
            logger.info("L'action n.25 n'exist pas ! Il faut faire attention !")
            pass

        return reward

    def _create_position_features(self):
        return np.array((self.broker.long_inventory.position_count / self.max_position,
                         self.broker.short_inventory.position_count / self.max_position,
                         self.broker.long_inventory.get_unrealized_pnl(self.midpoint),
                         self.broker.short_inventory.get_unrealized_pnl(self.midpoint),
                         self.broker.get_long_order_distance_to_midpoint(midpoint=self.midpoint),
                         self.broker.get_short_order_distance_to_midpoint(midpoint=self.midpoint)))

    def _create_action_features(self, action):
        return self.actions[action]

    def _create_bid_order_at_level(self, reward, discouragement, level=0):
        if level > 0:
            above_best_bid = self.bid_prices[self._local_step_number][level-1]
            best_bid = self.bid_prices[self._local_step_number][level]

            if round(above_best_bid, 2) == round(best_bid + 0.01, 2):
                price = round(self.midpoint - best_bid, 2)
                queue_ahead = self.bid_notionals[self._local_step_number][level]
            else:
                price = round(self.midpoint - best_bid + 0.01, 2)
                queue_ahead = 0.

            order = Order(ccy=self.sym, side='long', price=price,
                          step=self._local_step_number, queue_ahead=queue_ahead)
            if self.broker.add(order=order) is False:
                reward -= discouragement
        else:
            best_bid = self.bid_prices[self._local_step_number][level]
            price = round(self.midpoint - best_bid, 2)
            queue_ahead = self.bid_notionals[self._local_step_number][level]
            order = Order(ccy=self.sym, side='long', price=price,
                          step=self._local_step_number, queue_ahead=queue_ahead)
            if self.broker.add(order=order) is False:
                reward -= discouragement
        return reward

    def _create_ask_order_at_level(self, reward, discouragement, level=0):
        if level > 0:
            above_best_ask = self.ask_prices[self._local_step_number][level - 1]
            best_ask = self.ask_prices[self._local_step_number][level]

            if round(above_best_ask, 2) == round(best_ask - 0.01, 2):
                price = round(best_ask + self.midpoint, 2)
                queue_ahead = self.ask_notionals[self._local_step_number][level]
            else:
                price = round(best_ask + 0.01 + self.midpoint, 2)
                queue_ahead = 0.

            order = Order(ccy=self.sym, side='short', price=price,
                          step=self._local_step_number, queue_ahead=queue_ahead)
            if self.broker.add(order=order) is False:
                reward -= discouragement
        else:
            best_ask = self.ask_prices[self._local_step_number][level]
            price = round(best_ask + self.midpoint, 2)
            queue_ahead = self.ask_notionals[self._local_step_number][level]
            order = Order(ccy=self.sym, side='short', price=price,
                          step=self._local_step_number, queue_ahead=queue_ahead)
            if self.broker.add(order=order) is False:
                reward -= discouragement
        return reward


def _live_plotter(x_vec, y1_data, line1, identifier='Add Symbol Name', pause_time=0.00001):
    if not line1:
        # this is the call to matplotlib that allows dynamic plotting
        plt.ion()
        fig = plt.figure(figsize=(20, 12))
        ax = fig.add_subplot(111)
        # create a variable for the line so we can later update it
        line1, = ax.plot(x_vec, y1_data, '-', label='midpoint', alpha=0.8)
        # update plot label/title
        plt.ylabel('Price')
        plt.legend()
        plt.title('Title: {}'.format(identifier))
        plt.show(block=False)

    # after the figure, axis, and line are created, we only need to update the y-data
    line1.set_ydata(y1_data)

    # adjust limits if new data goes beyond bounds
    if np.min(y1_data) <= line1.axes.get_ylim()[0] or np.max(y1_data) >= line1.axes.get_ylim()[1]:
        plt.ylim(np.min(y1_data), np.max(y1_data))

    # this pauses the data so the figure/axis can catch up - the amount of pause can be altered above
    plt.pause(pause_time)

    # return line so we can update it again in the next iteration
    return line1
