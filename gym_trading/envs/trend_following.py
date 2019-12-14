from configurations import ENCOURAGEMENT
from gym_trading.envs.base_environment import BaseEnvironment
from gym_trading.utils.order import MarketOrder
from gym import spaces
import numpy as np


class TrendFollowing(BaseEnvironment):
    id = 'trend-following-v0'
    description = "Environment where agent can select market orders only"

    def __init__(self, **kwargs):
        """
        Environment designed to trade price jumps using market orders.

        :param kwargs: refer to BaseEnvironment.py
        """
        super(TrendFollowing, self).__init__(**kwargs)

        # environment attributes to override in sub-class
        self.actions = np.eye(3, dtype=np.float32)

        self.action_space = spaces.Discrete(len(self.actions))
        self.reset()  # reset to load observation.shape
        self.observation_space = spaces.Box(low=-10., high=10.,
                                            shape=self.observation.shape,
                                            dtype=np.float32)

        print('{} {} #{} instantiated\nobservation_space: {}'.format(
            TrendFollowing.id, self.symbol, self._seed, self.observation_space.shape),
            'reward_type = {}'.format(self.reward_type.upper()), 'max_steps = {}'.format(
                self.max_steps))

    def __str__(self):
        return '{} | {}-{}'.format(TrendFollowing.id, self.symbol, self._seed)

    def map_action_to_broker(self, action: int) -> (float, float):
        """
        Create or adjust orders per a specified action and adjust for penalties.

        :param action: (int) current step's action
        :return: (float) reward
        """
        reward = pnl = 0.0

        if action == 0:  # do nothing
            reward += ENCOURAGEMENT

        elif action == 1:  # buy
            if self.broker.short_inventory_count > 0:
                order = MarketOrder(ccy=self.symbol, side='short', price=self.midpoint,
                                    step=self.local_step_number)
                pnl += self.broker.remove(order=order)

            elif self.broker.long_inventory_count >= 0:
                order = MarketOrder(ccy=self.symbol, side='long', price=self.midpoint,
                                    step=self.local_step_number)
                if self.broker.add(order=order) is False:
                    reward -= ENCOURAGEMENT

            else:
                raise ValueError(('gym_trading.get_reward() Error for action #{} - '
                                  'unable to place an order with broker').format(action))

        elif action == 2:  # sell
            if self.broker.long_inventory_count > 0:
                order = MarketOrder(ccy=self.symbol, side='long', price=self.midpoint,
                                    step=self.local_step_number)
                pnl += self.broker.remove(order=order)

            elif self.broker.short_inventory_count >= 0:
                order = MarketOrder(ccy=self.symbol, side='short', price=self.midpoint,
                                    step=self.local_step_number)
                if self.broker.add(order=order) is False:
                    reward -= ENCOURAGEMENT

            else:
                raise ValueError(('gym_trading.get_reward() '
                                 'Error for action #{} - '
                                  'unable to place an order with broker').format(action))

        else:
            raise ValueError(('Unknown action to take in get_reward(): '
                              'action={} | midpoint={}').format(action, self.midpoint))

        return reward, pnl

    def _create_position_features(self) -> np.ndarray:
        """
        Create an array with features related to the agent's inventory.

        :return: (np.array) normalized position features
        """
        return np.array((self.broker.long_inventory.position_count / self.max_position,
                         self.broker.short_inventory.position_count / self.max_position,
                         self.broker.realized_pnl * self.broker.reward_scale,
                         self.broker.get_unrealized_pnl(self.best_bid, self.best_ask)
                        * self.broker.reward_scale),
                        dtype=np.float32)
