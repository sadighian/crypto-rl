from gym import Env
from abc import ABC, abstractmethod
from data_recorder.database.simulator import Simulator as Sim
from gym_trading.utils.render_env import TradingGraph
from configurations.configs import INDICATOR_WINDOW, INDICATOR_WINDOW_MAX
from gym_trading.indicators import RSI, TnS
from gym_trading.indicators.indicator import IndicatorManager
import numpy as np


class BaseEnvironment(Env, ABC):
    metadata = {'render.modes': ['human']}

    # Index of specific data points used to generate the observation space
    # Turn to true if Bitifinex is in the dataset (e.g., include_bitfinex=True)
    features = Sim.get_feature_labels(include_system_time=False, include_bitfinex=False)
    best_bid_index = features.index('coinbase_bid_distance_0')
    best_ask_index = features.index('coinbase_ask_distance_0')
    notional_bid_index = features.index('coinbase_bid_notional_0')
    notional_ask_index = features.index('coinbase_ask_notional_0')
    buy_trade_index = features.index('coinbase_buys')
    sell_trade_index = features.index('coinbase_sells')

    # Constants for scaling data
    target_pnl = 0.03  # 3.0% gain per episode (i.e., day)

    def __init__(self, fitting_file='LTC-USD_2019-04-07.csv.xz',
                 testing_file='LTC-USD_2019-04-08.csv.xz', step_size=1, max_position=5,
                 window_size=10, seed=1, action_repeats=10, training=True,
                 format_3d=False, z_score=True, reward_type='trade_completion',
                 scale_rewards=True):
        """
        Base class for creating environments extending OpenAI's GYM framework.

        :param fitting_file: historical data used to fit environment data (i.e.,
            previous trading day)
        :param testing_file: historical data used in environment
        :param step_size: increment size for steps (NOTE: leave a 1, otherwise market
            transaction data will be overlooked)
        :param max_position: maximum number of positions able to hold in inventory
        :param window_size: number of lags to include in observation space
        :param seed: random seed number
        :param action_repeats: number of steps to take in environment after a given action
        :param training: if TRUE, then randomize starting point in environment
        :param format_3d: if TRUE, reshape observation space from matrix to tensor
        :param z_score: if TRUE, normalize data set with Z-Score, otherwise use Min-Max
            (i.e., range of 0 to 1)
        :param reward_type: method for calculating the environment's reward:
            1) 'trade_completion' --> reward is generated per trade's round trip
            2) 'continuous_total_pnl' --> change in realized & unrealized pnl between
                                            time steps
            3) 'continuous_realized_pnl' --> change in realized pnl between time steps
            4) 'continuous_unrealized_pnl' --> change in unrealized pnl between time steps
        """
        # properties required for instantiation
        self.action_repeats = action_repeats
        self._seed = seed
        self._random_state = np.random.RandomState(seed=self._seed)
        self.training = training
        self.step_size = step_size
        self.max_position = max_position
        self.window_size = window_size
        self.reward_type = reward_type
        self.format_3d = format_3d  # e.g., [window, features, *NEW_AXIS*]
        self.sym = testing_file[:7]  # slice the CCY from the filename
        self.scale_rewards = scale_rewards

        # properties that get reset()
        self.reward = 0.0
        self.done = False
        self.local_step_number = 0
        self.midpoint = 0.0
        self.observation = None
        self.action = 0
        self.last_pnl = 0.

        # properties to override in sub-classes
        self.actions = None
        self.broker = None
        self.action_space = None
        self.observation_space = None

        # get historical data for simulations
        self.sim = Sim(use_arctic=False, z_score=z_score)

        self.prices_, self.data, self.normalized_data = self.sim.load_environment_data(
            fitting_file, testing_file)
        self.best_bid = self.best_ask = None

        self.max_steps = self.data.shape[0] - self.step_size * self.action_repeats - 1

        # load indicators into the indicator manager
        self.tns = IndicatorManager()
        self.rsi = IndicatorManager()
        for window in INDICATOR_WINDOW:
            self.tns.add(('tns_{}'.format(window), TnS(window=window)))
            self.rsi.add(('rsi_{}'.format(window), RSI(window=window)))

        # rendering class
        self._render = TradingGraph(sym=self.sym)

        # graph midpoint prices
        self._render.reset_render_data(
            y_vec=self.prices_[:np.shape(self._render.x_vec)[0]])

        # buffer for appending lags
        self.data_buffer = list()

    @abstractmethod
    def map_action_to_broker(self, action: int):
        """
        Translate agent's action into an order and submit order to broker.
        :param action: (int) agent's action for current step
        :return: (tuple) reward, pnl
        """
        return 0., 0.

    @abstractmethod
    def _create_position_features(self):
        """
        Create agent space feature set reflecting the positions held in inventory.
        :return: (np.array) position features
        """
        return np.array([np.nan], dtype=np.float32)

    def _get_step_reward(self, step_pnl: float):
        """
        Get reward for current time step.
            Note: 'reward_type' is set during environment instantiation.
        :param step_pnl: (float) PnL accrued from order fills at current time step
        :return: (float) reward
        """
        reward = 0.
        if self.reward_type == 'trade_completion':
            reward += step_pnl
            # Note: we do not need to update last_pnl for this reward approach
        elif self.reward_type == 'continuous_total_pnl':
            new_pnl = self.broker.get_total_pnl(self.best_bid, self.best_ask)
            reward += new_pnl - self.last_pnl  # Difference in PnL
            self.last_pnl = new_pnl
        elif self.reward_type == 'continuous_realized_pnl':
            new_pnl = self.broker.realized_pnl
            reward += new_pnl - self.last_pnl  # Difference in PnL
            self.last_pnl = new_pnl
        elif self.reward_type == 'continuous_unrealized_pnl':
            new_pnl = self.broker.get_unrealized_pnl(self.best_bid, self.best_ask)
            reward += new_pnl - self.last_pnl  # Difference in PnL
            self.last_pnl = new_pnl
        else:
            print("_get_step_reward() Unknown reward_type: {}".format(self.reward_type))

        if self.scale_rewards:
            reward /= self.broker.reward_scale

        return reward

    def step(self, action: int):
        """
        Step through environment with action
        :param action: (int) action to take in environment
        :return: (tuple) observation, reward, is_done, and empty `dict`
        """
        for current_step in range(self.action_repeats):

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
            self.midpoint = self.prices_[self.local_step_number]

            # Pass current time step bid/ask prices to broker to calculate PnL,
            # or if any open orders are to be filled
            self.best_bid, self.best_ask = self._get_nbbo()
            buy_volume = self._get_book_data(BaseEnvironment.buy_trade_index)
            sell_volume = self._get_book_data(BaseEnvironment.sell_trade_index)

            # Update indicators
            self.tns.step(buys=buy_volume, sells=sell_volume)
            self.rsi.step(price=self.midpoint)

            # Get PnL from any filled LIMIT orders
            limit_pnl = self.broker.step_limit_order_pnl(bid_price=self.best_bid,
                                                         ask_price=self.best_ask,
                                                         buy_volume=buy_volume,
                                                         sell_volume=sell_volume,
                                                         step=self.local_step_number)

            # Get PnL from any filled MARKET orders AND action penalties for invalid
            # actions made by the agent for future discouragement
            step_reward, market_pnl = self.map_action_to_broker(action=step_action)
            step_pnl = limit_pnl + step_reward + market_pnl
            self.reward += self._get_step_reward(step_pnl=step_pnl)

            step_observation = self._get_step_observation(action=action)
            self.data_buffer.append(step_observation)

            if len(self.data_buffer) > self.window_size:
                del self.data_buffer[0]

            self.local_step_number += self.step_size

        self.observation = self._get_observation()

        if self.local_step_number > self.max_steps:
            self.done = True
            flatten_pnl = self.broker.flatten_inventory(self.best_bid, self.best_ask)
            self.reward += self._get_step_reward(step_pnl=flatten_pnl)

        return self.observation, self.reward, self.done, {}

    def reset(self):
        """
        Reset the environment.
        :return: (np.array) Observation at first step
        """
        if self.training:
            self.local_step_number = self._random_state.randint(
                low=0, high=self.data.shape[0] // 4)
        else:
            self.local_step_number = 0

        msg = ' {}-{} reset. Episode pnl: {:.4f} with {} trades. First step: {}'.format(
            self.sym, self._seed, self.broker.get_total_pnl(self.best_bid, self.best_ask),
            self.broker.total_trade_count, self.local_step_number)
        print(msg)

        self.reward = 0.0
        self.done = False
        self.broker.reset()
        self.data_buffer.clear()
        self.rsi.reset()
        self.tns.reset()

        for step in range(self.window_size + INDICATOR_WINDOW_MAX):
            self.midpoint = self.prices_[self.local_step_number]
            self.best_bid, self.best_ask = self._get_nbbo()

            step_buy_volume = self._get_book_data(BaseEnvironment.buy_trade_index)
            step_sell_volume = self._get_book_data(BaseEnvironment.sell_trade_index)
            self.tns.step(buys=step_buy_volume, sells=step_sell_volume)
            self.rsi.step(price=self.midpoint)

            step_observation = self._get_step_observation(action=0)
            self.data_buffer.append(step_observation)

            self.local_step_number += self.step_size
            if len(self.data_buffer) > self.window_size:
                del self.data_buffer[0]

        self.observation = self._get_observation()

        return self.observation

    def render(self, mode='human'):
        """
        Render midpoint prices
        :param mode: (str) flag for type of rendering. Only 'human' supported.
        :return: (void)
        """
        self._render.render(midpoint=self.midpoint, mode=mode)

    def close(self):
        """
        Free clear memory when closing environment
        :return: (void)
        """
        self.data = None
        self.normalized_data = None
        self.prices_ = None
        self.broker = None
        self.sim = None
        self.data_buffer = None
        self.tns = None
        self.rsi = None

    def seed(self, seed=1):
        """
        Set random seed in environment
        :param seed: (int) random seed number
        :return: (list) seed number in a list
        """
        self._random_state = np.random.RandomState(seed=seed)
        self._seed = seed
        return [seed]

    @staticmethod
    def _process_data(_next_state):
        """
        Reshape observation for function approximator
        :param _next_state: observation space
        :return: (np.array) clipped observation space
        """
        return _next_state.reshape((1, -1))

    def _create_action_features(self, action):
        """
        Create a features array for the current time step's action.
        :param action: (int) action number
        :return: (np.array) One-hot of current action
        """
        return self.actions[action]

    def _create_indicator_features(self):
        """
        Create features vector with environment indicators.
        :return: (np.array) Indicator values for current time step
        """
        return np.array((*self.tns.get_value(), *self.rsi.get_value()), dtype=np.float32)

    def _get_nbbo(self):
        """
        Get best bid and offer
        :return: (tuple) best bid and offer
        """
        best_bid = round(
            self.midpoint - self._get_book_data(BaseEnvironment.best_bid_index), 2)
        best_ask = round(
            self.midpoint + self._get_book_data(BaseEnvironment.best_ask_index), 2)
        return best_bid, best_ask

    def _get_book_data(self, index=0):
        """
        Return step 'n' of order book snapshot data
        :param index: (int) step 'n' to look up in order book snapshot history
        :return: (np.array) order book snapshot vector
        """
        return self.data[self.local_step_number][index]

    def _get_step_observation(self, action=0):
        """
        Current step observation, NOT including historical data.
        :param action: (int) current step action
        :return: (np.array) Current step observation
        """
        step_position_features = self._create_position_features()
        step_action_features = self._create_action_features(action=action)
        step_indicator_features = self._create_indicator_features()
        return np.concatenate((
            self._process_data(self.normalized_data[self.local_step_number]),
            step_indicator_features, step_position_features, step_action_features,
            np.array([self.reward])), axis=None)

    def _get_observation(self):
        """
        Current step observation, including historical data.

        If format_3d is TRUE: Expand the observation space from 2 to 3 dimensions.
        (note: This is necessary for conv nets in Baselines.)
        :return: (np.array) Observation state for current time step
        """
        observation = np.array(self.data_buffer, dtype=np.float32)
        if self.format_3d:
            observation = np.expand_dims(observation, axis=-1)
        return observation
