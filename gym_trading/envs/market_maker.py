import numpy as np
from gym import spaces
from typing import Tuple

from configurations import ENCOURAGEMENT
from gym_trading.envs.base_environment import BaseEnvironment
from gym_trading.utils.order import LimitOrder


class MarketMaker(BaseEnvironment):
    id = 'market-maker-v0'
    description = "Environment where limit orders are tethered to LOB price levels"

    def __init__(self, **kwargs):
        """
        Environment designed for automated market making.

        :param kwargs: refer to BaseEnvironment.py
        """
        super().__init__(**kwargs)

        # Environment attributes to override in sub-class
        self.actions = np.eye(17, dtype=np.float32)

        self.action_space = spaces.Discrete(len(self.actions))
        self.observation = self.reset()  # Reset to load observation.shape
        self.observation_space = spaces.Box(low=-10., high=10.,
                                            shape=self.observation.shape,
                                            dtype=np.float32)

        # Add the remaining labels for the observation space
        self.viz.observation_labels += ['Long Dist', 'Short Dist',
                                        'Bid Completion Ratio', 'Ask Completion Ratio']
        self.viz.observation_labels += [f'Action #{a}' for a in range(len(self.actions))]
        self.viz.observation_labels += ['Reward']

        print('{} {} #{} instantiated\nobservation_space: {}'.format(
            MarketMaker.id, self.symbol, self._seed, self.observation_space.shape),
            'reward_type = {}'.format(self.reward_type.upper()), 'max_steps = {}'.format(
                self.max_steps))

    def __str__(self):
        return '{} | {}-{}'.format(MarketMaker.id, self.symbol, self._seed)

    def map_action_to_broker(self, action: int) -> Tuple[float, float]:
        """
        Create or adjust orders per a specified action and adjust for penalties.

        :param action: (int) current step's action
        :return: (float) reward
        """
        action_penalty = pnl = 0.0

        if action == 0:  # do nothing
            action_penalty += ENCOURAGEMENT

        elif action == 1:
            action_penalty += self._create_order_at_level(level=0, side='long')
            action_penalty += self._create_order_at_level(level=4, side='short')

        elif action == 2:
            action_penalty += self._create_order_at_level(level=0, side='long')
            action_penalty += self._create_order_at_level(level=9, side='short')

        elif action == 3:
            action_penalty += self._create_order_at_level(level=0, side='long')
            action_penalty += self._create_order_at_level(level=14, side='short')

        elif action == 4:
            action_penalty += self._create_order_at_level(level=4, side='long')
            action_penalty += self._create_order_at_level(level=0, side='short')

        elif action == 5:
            action_penalty += self._create_order_at_level(level=4, side='long')
            action_penalty += self._create_order_at_level(level=4, side='short')

        elif action == 6:
            action_penalty += self._create_order_at_level(level=4, side='long')
            action_penalty += self._create_order_at_level(level=9, side='short')

        elif action == 7:
            action_penalty += self._create_order_at_level(level=4, side='long')
            action_penalty += self._create_order_at_level(level=14, side='short')

        elif action == 8:
            action_penalty += self._create_order_at_level(level=9, side='long')
            action_penalty += self._create_order_at_level(level=0, side='short')

        elif action == 9:
            action_penalty += self._create_order_at_level(level=9, side='long')
            action_penalty += self._create_order_at_level(level=4, side='short')

        elif action == 10:
            action_penalty += self._create_order_at_level(level=9, side='long')
            action_penalty += self._create_order_at_level(level=9, side='short')

        elif action == 11:
            action_penalty += self._create_order_at_level(level=9, side='long')
            action_penalty += self._create_order_at_level(level=14, side='short')

        elif action == 12:
            action_penalty += self._create_order_at_level(level=14, side='long')
            action_penalty += self._create_order_at_level(level=0, side='short')

        elif action == 13:
            action_penalty += self._create_order_at_level(level=14, side='long')
            action_penalty += self._create_order_at_level(level=4, side='short')

        elif action == 14:
            action_penalty += self._create_order_at_level(level=14, side='long')
            action_penalty += self._create_order_at_level(level=9, side='short')

        elif action == 15:
            action_penalty += self._create_order_at_level(level=14, side='long')
            action_penalty += self._create_order_at_level(level=14, side='short')

        elif action == 16:
            pnl += self.broker.flatten_inventory(self.best_bid, self.best_ask)

        else:
            raise ValueError("L'action n'exist pas !!! Il faut faire attention !!!")

        return action_penalty, pnl

    def _create_position_features(self) -> np.ndarray:
        """
        Create an array with features related to the agent's inventory.

        :return: (np.array) normalized position features
        """
        return np.array((self.broker.net_inventory_count / self.max_position,
                         self.broker.realized_pnl * self.broker.pct_scale,
                         self.broker.get_unrealized_pnl(self.best_bid, self.best_ask)
                         * self.broker.pct_scale,
                         self.broker.get_long_order_distance_to_midpoint(
                             midpoint=self.midpoint) * self.broker.pct_scale,
                         self.broker.get_short_order_distance_to_midpoint(
                             midpoint=self.midpoint) * self.broker.pct_scale,
                         *self.broker.get_queues_ahead_features()), dtype=np.float32)

    def _create_order_at_level(self, level: int, side: str) -> float:
        """
        Create a new order at a specified LOB level.

        :param level: (int) level in the limit order book
        :param side: (str) direction of trade e.g., 'long' or 'short'
        :return: (float) reward with penalties added
        """
        reward = 0.0
        if side == 'long':
            notional_index = self.notional_bid_index
            price_index = self.best_bid_index
        elif side == 'short':
            notional_index = self.notional_ask_index
            price_index = self.best_ask_index
        else:
            notional_index = price_index = None
        # get price data from numpy array
        price_level_price = self._get_book_data(index=price_index + level)
        # transform percentage into a hard number
        price_level_price = round(self.midpoint * (price_level_price + 1.), 2)
        price_level_queue = self._get_book_data(index=notional_index + level)
        # create a new order
        order = LimitOrder(ccy=self.symbol,
                           side=side,
                           price=price_level_price,
                           step=self.local_step_number,
                           queue_ahead=price_level_queue)
        # add a penalty or encouragement, depending if order is accepted
        if self.broker.add(order=order) is False:
            reward -= ENCOURAGEMENT
        else:
            reward += ENCOURAGEMENT
        return reward
