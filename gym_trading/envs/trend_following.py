import numpy as np
from gym import spaces
from typing import Tuple

from configurations import ENCOURAGEMENT, MARKET_ORDER_FEE
from gym_trading.envs.base_environment import BaseEnvironment
from gym_trading.utils.order import MarketOrder


class TrendFollowing(BaseEnvironment):
    id = 'trend-following-v0'
    description = "Environment where agent can select market orders only"

    def __init__(self, **kwargs):
        """
        Environment designed to trade price jumps using market orders.

        :param kwargs: refer to BaseEnvironment.py
        """
        super().__init__(**kwargs)

        # Environment attributes to override in sub-class
        self.actions = np.eye(3, dtype=np.float32)

        self.action_space = spaces.Discrete(len(self.actions))
        self.reset()  # Reset to load observation.shape
        self.observation_space = spaces.Box(low=-10., high=10.,
                                            shape=self.observation.shape,
                                            dtype=np.float32)

        # Add the remaining labels for the observation space
        self.viz.observation_labels += [f'Action #{a}' for a in range(len(self.actions))]
        self.viz.observation_labels += ['Reward']

        print('{} {} #{} instantiated\nobservation_space: {}'.format(
            TrendFollowing.id, self.symbol, self._seed, self.observation_space.shape),
            'reward_type = {}'.format(self.reward_type.upper()), 'max_steps = {}'.format(
                self.max_steps))

    def __str__(self):
        return '{} | {}-{}'.format(TrendFollowing.id, self.symbol, self._seed)

    def map_action_to_broker(self, action: int) -> Tuple[float, float]:
        """
        Create or adjust orders per a specified action and adjust for penalties.

        :param action: (int) current step's action
        :return: (float) reward
        """
        action_penalty_reward = pnl = 0.0

        if action == 0:  # do nothing
            action_penalty_reward += ENCOURAGEMENT

        elif action == 1:  # buy
            # Deduct transaction costs
            if self.broker.transaction_fee:
                pnl -= MARKET_ORDER_FEE

            if self.broker.short_inventory_count > 0:
                # Net out existing position
                order = MarketOrder(ccy=self.symbol, side='short', price=self.best_ask,
                                    step=self.local_step_number)
                pnl += self.broker.remove(order=order)

            elif self.broker.long_inventory_count >= 0:
                order = MarketOrder(ccy=self.symbol, side='long', price=self.best_ask,
                                    step=self.local_step_number)
                if self.broker.add(order=order) is False:
                    action_penalty_reward -= ENCOURAGEMENT

            else:
                raise ValueError(('gym_trading.get_reward() Error for action #{} - '
                                  'unable to place an order with broker').format(action))

        elif action == 2:  # sell
            # Deduct transaction costs
            if self.broker.transaction_fee:
                pnl -= MARKET_ORDER_FEE

            if self.broker.long_inventory_count > 0:
                # Net out existing position
                order = MarketOrder(ccy=self.symbol, side='long', price=self.best_bid,
                                    step=self.local_step_number)
                pnl += self.broker.remove(order=order)

            elif self.broker.short_inventory_count >= 0:
                order = MarketOrder(ccy=self.symbol, side='short', price=self.best_bid,
                                    step=self.local_step_number)
                if self.broker.add(order=order) is False:
                    action_penalty_reward -= ENCOURAGEMENT

            else:
                raise ValueError(('gym_trading.get_reward() Error for action #{} - '
                                  'unable to place an order with broker').format(action))

        else:
            raise ValueError(('Unknown action to take in get_reward(): '
                              'action={} | midpoint={}').format(action, self.midpoint))

        return action_penalty_reward, pnl

    def _create_position_features(self) -> np.ndarray:
        """
        Create an array with features related to the agent's inventory.

        :return: (np.array) normalized position features
        """
        return np.array((self.broker.net_inventory_count / self.max_position,
                         self.broker.realized_pnl * self.broker.pct_scale,
                         self.broker.get_unrealized_pnl(self.best_bid, self.best_ask)
                         * self.broker.pct_scale),
                        dtype=np.float32)
