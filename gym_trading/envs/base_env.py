from data_recorder.database.simulator import Simulator as Sim
from gym_trading.utils.render_env import TradingGraph
from configurations.configs import (INDICATOR_WINDOW, INDICATOR_WINDOW_MAX, EMA_ALPHA,
                                    MARKET_ORDER_FEE)
from indicators import RSI, TnS, IndicatorManager, PnlNorm
from gym import Env
from abc import ABC, abstractmethod
import numpy as np


class BaseEnvironment(Env, ABC):
    metadata = {'render.modes': ['human']}

    # Index of specific data points used to generate the observation space
    # Turn to true if Bitifinex is in the dataset (e.g., include_bitfinex=True)
    features = Sim.get_feature_labels(include_system_time=False, include_bitfinex=False,
                                      include_imbalances=True, include_ema=False,
                                      include_spread=True)
    best_bid_index = features.index('coinbase_bid_distance_0')
    best_ask_index = features.index('coinbase_ask_distance_0')
    notional_bid_index = features.index('coinbase_bid_notional_0')
    notional_ask_index = features.index('coinbase_ask_notional_0')
    buy_trade_index = features.index('coinbase_buys')
    sell_trade_index = features.index('coinbase_sells')

    def __init__(self, fitting_file='BTC-USD_2019-04-07.csv.xz',
                 testing_file='BTC-USD_2019-04-08.csv.xz', step_size=1, max_position=5,
                 window_size=10, seed=1, action_repeats=10, training=True,
                 format_3d=True, z_score=True, reward_type='default',
                 scale_rewards=True, ema_alpha=EMA_ALPHA):
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
            5) 'normed' --> refer to https://arxiv.org/abs/1804.04216v1
            6) 'div' --> reward is generated per trade's round trip divided by
                inventory count (again, refer to https://arxiv.org/abs/1804.04216v1)
            7) 'asymmetrical' --> extended version of *default* and enhanced
                with a reward for being filled above/below midpoint,
                and returns only negative rewards for Unrealized PnL to
                discourage long-term speculation.
            8) 'asymmetrical_adj' --> extended version of *default* and enhanced
                with a reward for being filled above/below midpoint,
                and weighted up/down unrealized returns.
            9) 'default' --> Pct change in Unrealized PnL + Realized PnL of
                respective time step.
        :param ema_alpha: decay factor for EMA, usually between 0.9 and 0.9999; if NONE,
            raw values are returned in place of smoothed values
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
        self.last_midpoint = None
        self.midpoint_change = None

        # properties to override in sub-classes
        self.actions = None
        self.broker = None
        self.action_space = None
        self.observation_space = None

        # get historical data for simulations
        self.sim = Sim(z_score=z_score, alpha=ema_alpha)

        self.prices_, self.data, self.normalized_data = self.sim.load_environment_data(
            fitting_file=fitting_file, testing_file=testing_file,
            include_imbalances=True, as_pandas=False)
        self.best_bid = self.best_ask = None

        self.max_steps = self.data.shape[0] - self.step_size * self.action_repeats - 1

        # load indicators into the indicator manager
        self.tns = IndicatorManager()
        self.rsi = IndicatorManager()
        for window in INDICATOR_WINDOW:
            self.tns.add(('tns_{}'.format(window), TnS(window=window, alpha=ema_alpha)))
            self.rsi.add(('rsi_{}'.format(window), RSI(window=window, alpha=ema_alpha)))

        # conditionally load PnlNorm, since it calculates in O(n) time complexity
        self.pnl_norm = PnlNorm(window=INDICATOR_WINDOW[0],
                                alpha=None) if self.reward_type == 'normed' else None

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

    @staticmethod
    def _trade_completion_reward(step_pnl: float):
        """
        Alternate approach for reward calculation which places greater importance on
        trades that have returned at least a 1:1 profit-to-loss ratio after
        transaction fees.
        :param step_pnl: limit order pnl and any penalties for bad actions
        :return: normalized reward (-0.1 to 0.1) range, which can be scaled to
            (-1, 1) in self._get_step_reward() method
        """
        reward = 0.0
        if step_pnl > MARKET_ORDER_FEE * 2:  # e.g.,  2:1 profit to loss ratio
            reward += 1.0
        elif step_pnl > 0.0:
            reward += step_pnl
        elif step_pnl < -MARKET_ORDER_FEE:  # skew penalty so
            reward -= 1.0
        else:
            reward -= step_pnl
        return reward

    def _asymmetrical_reward(self, long_filled: bool, short_filled: bool, step_pnl: float,
                             dampening=0.15):
        """
        Asymmetrical reward type for environments, which is derived from percentage
        changes and notional values.
        The inputs are as follows:
            (1) Change in exposure value between time steps, in percentage terms; and,
            (2) Realized PnL from a open order being filled between time steps,
                in dollar terms.
        :param long_filled: TRUE if long order is filled within same time step
        :param short_filled: TRUE if short order is filled within same time step
        :param step_pnl: limit order pnl and any penalties for bad actions
        :param dampening: discount factor towards pnl change between time steps
        :return: (float)
        """
        exposure_change = self.broker.total_inventory_count * self.midpoint_change
        long_fill_reward = short_fill_reward = 0.

        if long_filled:
            long_fill_reward += ((self.midpoint / self.best_bid) - 1.)
            print("long_fill_reward: {:.6f}".format(long_fill_reward))
        if short_filled:
            short_fill_reward += ((self.best_ask / self.midpoint) - 1.)
            print("short_fill_reward: {:.6f}".format(short_fill_reward))

        reward = (long_fill_reward + short_fill_reward) + \
            min(0., exposure_change * dampening)

        if long_filled:
            reward += step_pnl
        if short_filled:
            reward += step_pnl

        return reward

    def _asymmetrical_reward_adj(self, long_filled: bool, short_filled: bool,
                                 step_pnl: float, dampening=0.25):
        """
        Asymmetrical reward type for environments with balanced feedback, which is
        derived from percentage
        changes and notional values.
        The inputs are as follows:
            (1) Change in exposure value between time steps, in percentage terms; and,
            (2) Realized PnL from a open order being filled between time steps,
                in dollar terms.
        :param long_filled: TRUE if long order is filled within same time step
        :param short_filled: TRUE if short order is filled within same time step
        :param step_pnl: limit order pnl and any penalties for bad actions
        :param dampening: discount factor towards pnl change between time steps
        :return: (float)
        """
        exposure_change = self.broker.total_inventory_count * self.midpoint_change
        long_fill_reward = short_fill_reward = 0.

        if long_filled:
            long_fill_reward += ((self.midpoint / self.best_bid) - 1.)
            print("long_fill_reward: {:.6f}".format(long_fill_reward))
        if short_filled:
            short_fill_reward += ((self.best_ask / self.midpoint) - 1.)
            print("short_fill_reward: {:.6f}".format(short_fill_reward))

        reward = (long_fill_reward + short_fill_reward) + \
            min(0., exposure_change * (1. - dampening)*0.1) + \
            max(0., exposure_change * dampening*0.1)

        if long_filled:
            reward += step_pnl
        if short_filled:
            reward += step_pnl

        return reward

    def _default_reward(self, long_filled: bool, short_filled: bool, step_pnl: float):
        """
        Default reward type for environments, which is derived from PnL and order
        quantity.
        The inputs are as follows:
            (1) Change in exposure value between time steps, in dollar terms; and,
            (2) Realized PnL from a open order being filled between time steps,
                in dollar terms.
        :param long_filled: TRUE if long order is filled within same time step
        :param short_filled: TRUE if short order is filled within same time step
        :param step_pnl: limit order pnl and any penalties for bad actions
        :return:
        """
        reward = self.broker.total_inventory_count * self.midpoint_change
        if long_filled:
            reward += step_pnl
        if short_filled:
            reward += step_pnl
        return reward

    def _get_step_reward(self, step_pnl: float, long_filled: bool, short_filled: bool):
        """
        Get reward for current time step.
            Note: 'reward_type' is set during environment instantiation.
        :param step_pnl: (float) PnL accrued from order fills at current time step
        :return: (float) reward
        """
        reward = 0.0
        if self.reward_type == 'default':  # pnl in dollar terms
            reward += self._default_reward(long_filled, short_filled, step_pnl)
        elif self.reward_type == 'asymmetrical':
            reward += self._asymmetrical_reward(long_filled=long_filled,
                                                short_filled=short_filled,
                                                step_pnl=step_pnl)
        elif self.reward_type == 'asymmetrical_adj':
            reward += self._asymmetrical_reward_adj(long_filled=long_filled,
                                                    short_filled=short_filled,
                                                    step_pnl=step_pnl)
        elif self.reward_type == 'trade_completion':  # reward is [-1,1]
            reward += self._trade_completion_reward(step_pnl=step_pnl)
            # Note: we do not need to update last_pnl for this reward approach
        elif self.reward_type == 'continuous_total_pnl':  # pnl in percentage
            new_pnl = self.broker.get_total_pnl(self.best_bid, self.best_ask)
            difference = new_pnl - self.last_pnl  # Difference in PnL over time step
            # include step_pnl to net out drops in unrealized PnL from position closing
            reward += difference + step_pnl
            self.last_pnl = new_pnl
        elif self.reward_type == 'continuous_realized_pnl':
            new_pnl = self.broker.realized_pnl
            reward += new_pnl - self.last_pnl  # Difference in PnL
            self.last_pnl = new_pnl
        elif self.reward_type == 'continuous_unrealized_pnl':
            new_pnl = self.broker.get_unrealized_pnl(self.best_bid, self.best_ask)
            difference = new_pnl - self.last_pnl  # Difference in PnL over time step
            # include step_pnl to net out drops in unrealized PnL from position closing
            reward += difference + step_pnl
            self.last_pnl = new_pnl
        elif self.reward_type == 'normed':
            # refer to https://arxiv.org/abs/1804.04216v1
            new_pnl = self.pnl_norm.raw_value
            reward += new_pnl - self.last_pnl  # Difference in PnL
            self.last_pnl = new_pnl
        elif self.reward_type == 'div':
            reward += step_pnl / max(
                self.broker.total_inventory_count, 1)
        else:  # Default implementation
            reward += self._default_reward(long_filled, short_filled, step_pnl)

        if self.scale_rewards:
            reward *= 100.  # multiply to avoid division error

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
            self.midpoint_change = (self.midpoint / self.last_midpoint) - 1.

            # Pass current time step bid/ask prices to broker to calculate PnL,
            # or if any open orders are to be filled
            self.best_bid, self.best_ask = self._get_nbbo()
            buy_volume = self._get_book_data(BaseEnvironment.buy_trade_index)
            sell_volume = self._get_book_data(BaseEnvironment.sell_trade_index)

            # Update indicators
            self.tns.step(buys=buy_volume, sells=sell_volume)
            self.rsi.step(price=self.midpoint)

            # Get PnL from any filled LIMIT orders
            limit_pnl, long_filled, short_filled = self.broker.step_limit_order_pnl(
                bid_price=self.best_bid, ask_price=self.best_ask, buy_volume=buy_volume,
                sell_volume=sell_volume, step=self.local_step_number)

            # Get PnL from any filled MARKET orders AND action penalties for invalid
            # actions made by the agent for future discouragement
            step_reward, market_pnl = self.map_action_to_broker(action=step_action)
            step_pnl = limit_pnl + step_reward + market_pnl

            # step thru pnl_norm if not None
            if self.pnl_norm:
                self.pnl_norm.step(
                    pnl=self.broker.get_unrealized_pnl(
                        bid_price=self.best_bid,
                        ask_price=self.best_ask))

            self.reward += self._get_step_reward(step_pnl=step_pnl,
                                                 long_filled=long_filled,
                                                 short_filled=short_filled)

            step_observation = self._get_step_observation(action=action)
            self.data_buffer.append(step_observation)

            if len(self.data_buffer) > self.window_size:
                del self.data_buffer[0]

            self.local_step_number += self.step_size
            self.last_midpoint = self.midpoint

        self.observation = self._get_observation()

        if self.local_step_number > self.max_steps:
            self.done = True
            flatten_pnl = self.broker.flatten_inventory(self.best_bid, self.best_ask)
            self.reward += self._get_step_reward(step_pnl=flatten_pnl,
                                                 long_filled=False,
                                                 short_filled=False)

        return self.observation, self.reward, self.done, {}

    def reset(self):
        """
        Reset the environment.
        :return: (np.array) Observation at first step
        """
        if self.training:
            self.local_step_number = self._random_state.randint(
                low=0, high=self.data.shape[0] // 5)
        else:
            self.local_step_number = 0

        msg = (
            ' {}-{} reset. Episode pnl: {:.4f} with {} trades. '
            'Avg. Trade PnL: {:.4f}.  First step: {}').format(
            self.sym, self._seed, self.broker.realized_pnl, self.broker.total_trade_count,
            self.broker.average_trade_pnl, self.local_step_number
        )
        print(msg)

        self.reward = 0.0
        self.done = False
        self.broker.reset()
        self.data_buffer.clear()
        self.rsi.reset()
        self.tns.reset()
        if self.pnl_norm:
            self.pnl_norm.reset()

        for step in range(self.window_size + INDICATOR_WINDOW_MAX + 1):
            self.midpoint = self.prices_[self.local_step_number]
            self.best_bid, self.best_ask = self._get_nbbo()

            step_buy_volume = self._get_book_data(BaseEnvironment.buy_trade_index)
            step_sell_volume = self._get_book_data(BaseEnvironment.sell_trade_index)
            self.tns.step(buys=step_buy_volume, sells=step_sell_volume)
            self.rsi.step(price=self.midpoint)

            # step thru pnl_norm if not None
            if self.pnl_norm:
                self.pnl_norm.step(
                    pnl=self.broker.get_unrealized_pnl(
                        bid_price=self.best_bid, ask_price=self.best_ask))

            step_observation = self._get_step_observation(action=0)
            self.data_buffer.append(step_observation)

            self.local_step_number += self.step_size
            self.last_midpoint = self.midpoint
            if len(self.data_buffer) > self.window_size:
                del self.data_buffer[0]

        self.midpoint_change = (self.midpoint / self.last_midpoint) - 1.
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
        self.broker.reset()
        self.data_buffer.clear()
        self.sim = None
        self.tns = None
        self.rsi = None
        self.pnl_norm = None

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
        return np.clip(_next_state.reshape((1, -1)), -10, 10)

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
        return np.array((*self.tns.get_value(), *self.rsi.get_value()),
                        dtype=np.float32).reshape(1, -1)

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
            np.array([self.reward], dtype=np.float32)),
            axis=None)

    def _get_observation(self):
        """
        Current step observation, including historical data.

        If format_3d is TRUE: Expand the observation space from 2 to 3 dimensions.
        (note: This is necessary for conv nets in Baselines.)
        :return: (np.array) Observation state for current time step
        """
        # Note: reversing the data to chronological order is actually faster when
        # making an array in Python / Numpy, which is odd. #timeit
        observation = np.asarray(self.data_buffer, dtype=np.float32)
        if self.format_3d:
            observation = np.expand_dims(observation, axis=-1)
        return observation
