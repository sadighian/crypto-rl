from configurations import (
    INDICATOR_WINDOW, INDICATOR_WINDOW_MAX, EMA_ALPHA,
    MARKET_ORDER_FEE,
)
from indicators import IndicatorManager, RSI, TnS
from gym_trading.utils.render_env import TradingGraph
from gym_trading.utils.plot_history import Visualize
from gym_trading.utils.statistics import ExperimentStatistics
from gym_trading.utils.broker import Broker
from gym_trading.utils.data_pipeline import DataPipeline
import gym_trading.utils.reward as reward_types

from gym import Env
from abc import ABC, abstractmethod
from collections import deque
import pandas as pd
import numpy as np

VALID_REWARD_TYPES = [f for f in dir(reward_types) if '__' not in f]


class BaseEnvironment(Env, ABC):
    metadata = {'render.modes': ['human']}

    def __init__(self,
                 symbol: str,
                 fitting_file: str,
                 testing_file: str,
                 max_position: int = 10,
                 window_size: int = 5,
                 seed: int = 1,
                 action_repeats: int = 5,
                 training: bool = True,
                 format_3d: bool = True,
                 reward_type: str = 'default',
                 transaction_fee: bool = True,
                 ema_alpha: list or float or None = EMA_ALPHA):
        """
        Base class for creating environments extending OpenAI's GYM framework.

        :param symbol: currency pair to trade / experiment
        :param fitting_file: prior trading day (e.g., T-1)
        :param testing_file: current trading day (e.g., T)
        :param max_position: maximum number of positions able to hold in inventory
        :param window_size: number of lags to include in observation space
        :param seed: random seed number
        :param action_repeats: number of steps to take in environment after a given action
        :param training: if TRUE, then randomize starting point in environment
        :param format_3d: if TRUE, reshape observation space from matrix to tensor
        :param reward_type: method for calculating the environment's reward:
            1) 'default' --> inventory count * change in midpoint price returns
            2) 'default_with_fills' --> inventory count * change in midpoint price returns
                + closed trade PnL
            3) 'realized_pnl' --> change in realized pnl between time steps
            4) 'differential_sharpe_ratio' -->
        http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.1.7210&rep=rep1&type=pdf
            5) 'asymmetrical' --> extended version of *default* and enhanced with a
                    reward for being filled above or below midpoint, and returns only
                    negative rewards for Unrealized PnL to discourage long-term
                    speculation.
            6) 'trade_completion' --> reward is generated per trade's round trip

        :param ema_alpha: decay factor for EMA, usually between 0.9 and 0.9999; if NONE,
            raw values are returned in place of smoothed values
        """
        assert reward_type in VALID_REWARD_TYPES, \
            'Error: {} is not a valid reward type. Value must be in:\n{}'.format(
                reward_type, VALID_REWARD_TYPES)

        self.viz = Visualize(
            columns=['midpoint', 'buys', 'sells', 'inventory', 'realized_pnl'],
            store_historical_observations=True)

        # get Broker class to keep track of PnL and orders
        self.broker = Broker(max_position=max_position, transaction_fee=transaction_fee)

        # properties required for instantiation
        self.symbol = symbol
        self.action_repeats = action_repeats
        self._seed = seed
        self._random_state = np.random.RandomState(seed=self._seed)
        self.training = training
        self.max_position = max_position
        self.window_size = window_size
        self.reward_type = reward_type
        self.format_3d = format_3d  # e.g., [window, features, *NEW_AXIS*]
        self.testing_file = testing_file

        # properties that get reset()
        self.reward = np.array([0.0], dtype=np.float32)
        self.done = False
        self.local_step_number = 0
        self.midpoint = 0.0
        self.observation = None
        self.action = 0
        self.last_pnl = 0.
        self.last_midpoint = None
        self.midpoint_change = None
        self.A_t, self.B_t = 0., 0.  # variables for Differential Sharpe Ratio
        self.episode_stats = ExperimentStatistics()

        # properties to override in sub-classes
        self.actions = None
        self.action_space = None
        self.observation_space = None

        # get historical data for simulations
        self.data_pipeline = DataPipeline(alpha=ema_alpha)

        # three different data sets, for different purposes:
        #   1) midpoint_prices - midpoint prices that have not been transformed
        #   2) raw_data - raw limit order book data, not including imbalances
        #   3) normalized_data - z-scored limit order book and order flow imbalance
        #       data, also midpoint price feature is replace by midpoint log price change
        self._midpoint_prices, self._raw_data, self._normalized_data = \
            self.data_pipeline.load_environment_data(
                fitting_file=fitting_file,
                testing_file=testing_file,
                include_imbalances=True,
                as_pandas=True,
            )
        self.best_bid = self.best_ask = None

        self.max_steps = self._raw_data.shape[0] - self.action_repeats - 1

        # load indicators into the indicator manager
        self.tns = IndicatorManager()
        self.rsi = IndicatorManager()
        for window in INDICATOR_WINDOW:
            self.tns.add(('tns_{}'.format(window), TnS(window=window, alpha=ema_alpha)))
            self.rsi.add(('rsi_{}'.format(window), RSI(window=window, alpha=ema_alpha)))

        # buffer for appending lags
        self.data_buffer = deque(maxlen=self.window_size)

        # Index of specific data points used to generate the observation space
        features = self._raw_data.columns.tolist()
        self.best_bid_index = features.index('bid_distance_0')
        self.best_ask_index = features.index('ask_distance_0')
        self.notional_bid_index = features.index('bid_notional_0')
        self.notional_ask_index = features.index('ask_notional_0')
        self.buy_trade_index = features.index('buys')
        self.sell_trade_index = features.index('sells')

        # typecast all data sets to numpy
        self._raw_data = self._raw_data.to_numpy(dtype=np.float32)
        self._normalized_data = self._normalized_data.to_numpy(dtype=np.float32)
        self._midpoint_prices = self._midpoint_prices.to_numpy(dtype=np.float64)

        # rendering class
        self._render = TradingGraph(sym=self.symbol)

        # graph midpoint prices
        self._render.reset_render_data(
            y_vec=self._midpoint_prices[:np.shape(self._render.x_vec)[0]])

    @abstractmethod
    def map_action_to_broker(self, action: int) -> (float, float):
        """
        Translate agent's action into an order and submit order to broker.

        :param action: (int) agent's action for current step
        :return: (tuple) reward, pnl
        """
        return 0., 0.

    @abstractmethod
    def _create_position_features(self) -> np.ndarray:
        """
        Create agent space feature set reflecting the positions held in inventory.

        :return: (np.array) position features
        """
        return np.array([np.nan], dtype=np.float32)

    def _get_step_reward(self,
                         step_pnl: float,
                         step_penalty: float,
                         long_filled: bool,
                         short_filled: bool) -> float:
        """
        Calculate current step reward using a reward function.

        :param step_pnl: PnL realized from an open position that's been closed in the
        current time step
        :param step_penalty: Penalty signal for agent to discourage erroneous actions
        :param long_filled: TRUE if open long limit order was filled in current time step
        :param short_filled: TRUE if open short limit order was filled in current time
        step
        :return: reward for current time step
        """
        reward = 0.

        if self.reward_type == 'default':
            reward += reward_types.default(
                inventory_count=self.broker.total_inventory_count,
                midpoint_change=self.midpoint_change,
                step_penalty=step_penalty) * 100.

        elif self.reward_type == 'default_with_fills':
            reward += reward_types.default_with_fills(
                inventory_count=self.broker.total_inventory_count,
                midpoint_change=self.midpoint_change,
                step_pnl=step_pnl,
                step_penalty=step_penalty) * 100.

        elif self.reward_type == 'asymmetrical':
            reward += reward_types.asymmetrical(
                inventory_count=self.broker.total_inventory_count,
                midpoint_change=self.midpoint_change,
                half_spread_pct=(self.midpoint / self.best_bid) - 1.,
                long_filled=long_filled,
                short_filled=short_filled,
                step_pnl=step_pnl,
                dampening=0.3) * 100.

        elif self.reward_type == 'realized_pnl':
            current_pnl = self.broker.realized_pnl
            reward += reward_types.realized_pnl(
                current_pnl=current_pnl,
                last_pnl=self.last_pnl,
                step_penalty=step_penalty) * 100.
            self.last_pnl = current_pnl

        elif self.reward_type == 'differential_sharpe_ratio':
            # current time step PnL
            # current_pnl = self.broker.get_unrealized_pnl(*self._get_nbbo())
            # calculate Differential Sharpe Ratio
            tmp_reward, self.A_t, self.B_t = reward_types.differential_sharpe_ratio(
                R_t=self.midpoint_change * self.broker.total_inventory_count,
                A_tm1=self.A_t,
                B_tm1=self.B_t)
            reward += tmp_reward

            # updated last_pnl for the proceeding step
            # self.last_pnl = current_pnl

        elif self.reward_type == 'trade_completion':
            reward += reward_types.trade_completion(
                step_pnl=step_pnl,
                market_order_fee=MARKET_ORDER_FEE,
                profit_ratio=2.)

        else:  # Default implementation
            reward += reward_types.default(
                inventory_count=self.broker.total_inventory_count,
                midpoint_change=self.midpoint_change,
                step_penalty=step_penalty) * 100.

        return reward

    def step(self, action: int = 0) -> (np.ndarray, np.ndarray, bool, dict):
        """
        Step through environment with action.

        :param action: (int) action to take in environment
        :return: (tuple) observation, reward, is_done, and empty `dict`
        """
        for current_step in range(self.action_repeats):

            if self.done:
                self.reset()
                return self.observation, self.reward, self.done

            if current_step == 0:
                # reset the reward on the first step
                self.reward = 0.
                step_action = action
            else:
                # accumulate rewards on steps proceeding the first
                step_action = 0

            # Get current step's midpoint and change in midpoint price percentage
            self.midpoint = self._midpoint_prices[self.local_step_number]
            self.midpoint_change = (self.midpoint / self.last_midpoint) - 1.

            # Pass current time step bid/ask prices to broker to calculate PnL,
            # or if any open orders are to be filled
            self.best_bid, self.best_ask = self._get_nbbo()

            # verify the data integrity
            assert self.best_bid <= self.best_ask, (
                "Error: best bid is more expensive than the best Ask:"
                "\nBid = {}\nAsk = {}").format(self.best_bid, self.best_ask)

            # get buy and sell trade volume to use by indicators and 'broker' to
            # execute any open orders the agent has
            buy_volume = self._get_book_data(index=self.buy_trade_index)
            sell_volume = self._get_book_data(index=self.sell_trade_index)

            # Update indicators
            self.tns.step(buys=buy_volume, sells=sell_volume)
            self.rsi.step(price=self.midpoint)

            # Get PnL from any filled LIMIT orders, which is calculated by netting out
            # whatever open position the agent already has in FIFO order
            limit_pnl, long_filled, short_filled = self.broker.step_limit_order_pnl(
                bid_price=self.best_bid,
                ask_price=self.best_ask,
                buy_volume=buy_volume,
                sell_volume=sell_volume,
                step=self.local_step_number
            )

            # Get PnL from any filled MARKET orders AND action penalties for invalid
            # actions made by the agent for future discouragement
            action_penalty_reward, market_pnl = self.map_action_to_broker(
                action=step_action)
            # combine flatten_order action PnL with filled limit orders to derive the
            # total PnL generated in the current step
            step_pnl = limit_pnl + market_pnl
            self.reward += self._get_step_reward(step_pnl=step_pnl,
                                                 step_penalty=action_penalty_reward,
                                                 long_filled=long_filled,
                                                 short_filled=short_filled)

            step_observation = self._get_step_observation(action=action)
            self.viz.add_observation(obs=step_observation)
            self.data_buffer.append(step_observation)

            # store for visualization AFTER the episode
            self.viz.add(self.midpoint,  # arguments map to the column names in _init_
                         int(long_filled),
                         int(short_filled),
                         self.broker.long_inventory_count -
                         self.broker.short_inventory_count,
                         # normalize PnL by the max num of positions, thereby assuming
                         # 1:1 leverage. Also, multiply by 100 for readability
                         (self.broker.realized_pnl * 100) / self.max_position)

            self.local_step_number += 1
            self.last_midpoint = self.midpoint

        self.observation = self._get_observation()

        if self.local_step_number > self.max_steps:
            self.done = True

            had_long_positions = self.broker.long_inventory_count > 0
            had_short_positions = self.broker.short_inventory_count > 0

            flatten_pnl = self.broker.flatten_inventory(bid_price=self.best_bid,
                                                        ask_price=self.best_ask)
            self.reward += self._get_step_reward(step_pnl=flatten_pnl,
                                                 step_penalty=0.,
                                                 long_filled=False,
                                                 short_filled=False)

            # store for visualization AFTER the episode
            self.viz.add(self.midpoint,  # arguments map to the column names in _init_
                         int(had_long_positions),
                         int(had_short_positions),
                         self.broker.long_inventory_count -
                         self.broker.short_inventory_count,
                         # normalize PnL by the max num of positions, thereby assuming
                         # 1:1 leverage. Also, multiply by 100 for readability
                         (self.broker.realized_pnl * 100) / self.max_position)

        # save rewards to derive cumulative reward
        self.episode_stats.reward += self.reward

        return self.observation, self.reward, self.done, {}

    def reset(self) -> np.ndarray:
        """
        Reset the environment.

        :return: (np.array) Observation at first step
        """
        if self.training:
            self.local_step_number = self._random_state.randint(low=0,
                                                                high=self.max_steps // 5)
        else:
            self.local_step_number = 0

        # print out episode statistics if there was any activity by the agent
        if self.broker.total_trade_count > 0 or self.broker.realized_pnl != 0.:
            self.episode_stats.number_of_episodes += 1
            print(('-' * 25), '{}-{} {} EPISODE RESET'.format(
                self.symbol, self._seed, self.reward_type.upper()), ('-' * 25))
            print('Episode Reward: {:.4f}'.format(self.episode_stats.reward))
            print('Episode PnL: {:.2f}%'.format(
                (self.broker.realized_pnl / self.max_position) * 100.))
            print('Trade Count: {}'.format(self.broker.total_trade_count))
            print('Average PnL per Trade: {:.4f}%'.format(
                self.broker.average_trade_pnl * 100.))
            print('Total # of episodes: {}'.format(self.episode_stats.number_of_episodes))
            print('\n'.join(['{}\t=\t{}'.format(k, v) for k, v in
                             self.broker.get_statistics().items()]))
            print('First step:\t{}'.format(self.local_step_number))
            print(('-' * 70))
        else:
            print('Resetting environment #{} on episode #{}.'.format(
                self._seed, self.episode_stats.number_of_episodes))

        self.A_t, self.B_t = 0., 0.
        self.reward = 0.0
        self.done = False
        self.broker.reset()
        self.data_buffer.clear()
        self.episode_stats.reset()
        self.rsi.reset()
        self.tns.reset()
        self.viz.reset()

        for step in range(self.window_size + INDICATOR_WINDOW_MAX + 1):
            self.midpoint = self._midpoint_prices[self.local_step_number]

            if self.last_midpoint is None:
                self.last_midpoint = self.midpoint

            self.midpoint_change = (self.midpoint / self.last_midpoint) - 1.
            self.best_bid, self.best_ask = self._get_nbbo()
            step_buy_volume = self._get_book_data(index=self.buy_trade_index)
            step_sell_volume = self._get_book_data(index=self.sell_trade_index)
            self.tns.step(buys=step_buy_volume, sells=step_sell_volume)
            self.rsi.step(price=self.midpoint)

            step_observation = self._get_step_observation(action=0)
            self.data_buffer.append(step_observation)

            self.local_step_number += 1
            self.last_midpoint = self.midpoint

        self.observation = self._get_observation()

        return self.observation

    def render(self, mode: str = 'human') -> None:
        """
        Render midpoint prices.

        :param mode: (str) flag for type of rendering. Only 'human' supported.
        :return: (void)
        """
        self._render.render(midpoint=self.midpoint, mode=mode)

    def close(self) -> None:
        """
        Free clear memory when closing environment.

        :return: (void)
        """
        self.broker.reset()
        self.data_buffer.clear()
        self.episode_stats = None
        self._raw_data = None
        self._normalized_data = None
        self._midpoint_prices = None
        self.tns = None
        self.rsi = None

    def seed(self, seed: int = 1) -> list:
        """
        Set random seed in environment.

        :param seed: (int) random seed number
        :return: (list) seed number in a list
        """
        self._random_state = np.random.RandomState(seed=seed)
        self._seed = seed
        return [seed]

    @staticmethod
    def _process_data(observation: np.ndarray) -> np.ndarray:
        """
        Reshape observation for function approximator.

        :param observation: observation space
        :return: (np.array) clipped observation space
        """
        return np.clip(observation, -10., 10.)

    def _create_action_features(self, action: int) -> np.ndarray:
        """
        Create a features array for the current time step's action.

        :param action: (int) action number
        :return: (np.array) One-hot of current action
        """
        return self.actions[action]

    def _create_indicator_features(self) -> np.ndarray:
        """
        Create features vector with environment indicators.

        :return: (np.array) Indicator values for current time step
        """
        return np.array((*self.tns.get_value(),
                         *self.rsi.get_value()),
                        dtype=np.float32).reshape(1, -1)

    def _get_nbbo(self) -> (float, float):
        """
        Get best bid and offer.

        :return: (tuple) best bid and offer
        """
        best_bid = round(
            self.midpoint * (self._get_book_data(index=self.best_bid_index) + 1.), 2)
        best_ask = round(
            self.midpoint * (self._get_book_data(index=self.best_ask_index) + 1.), 2)
        return best_bid, best_ask

    def _get_book_data(self, index: int = 0) -> float or np.ndarray:
        """
        Return step 'n' of order book snapshot data.

        :param index: step 'n' to look up in order book snapshot history
        :return: order book snapshot vector
        """
        return self._raw_data[self.local_step_number][index]

    def _get_step_observation(self, action: int = 0) -> np.ndarray:
        """
        Current step observation, NOT including historical data.

        :param action: (int) current step action
        :return: (np.array) Current step observation
        """
        step_position_features = self._create_position_features()
        step_action_features = self._create_action_features(action=action)
        step_indicator_features = self._create_indicator_features()
        return self._process_data(np.concatenate((
            self._normalized_data[self.local_step_number],
            step_indicator_features,
            step_position_features,
            step_action_features,
            self.reward),
            axis=None)
        )

    def _get_observation(self) -> np.ndarray:
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

    def get_trade_history(self) -> pd.DataFrame:
        """
        Get DataFrame with trades from most recent episode.

        :return: midpoint prices, and buy & sell trades
        """
        return self.viz.to_df()

    def plot_trade_history(self) -> None:
        """
        Plot history from back-test with trade executions, total inventory, and PnL.

        :return:
        """
        self.viz.plot()

    def plot_observation_history(self) -> None:
        """
        Plot observation space as an image.

        :return:
        """
        return self.viz.plot_obs()
