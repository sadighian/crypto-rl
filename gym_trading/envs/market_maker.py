from configurations.configs import LIMIT_ORDER_FEE
from gym_trading.envs.base_env import BaseEnvironment
from gym_trading.utils.order import LimitOrder
from gym_trading.utils.broker import Broker
from gym import spaces
import numpy as np


class MarketMaker(BaseEnvironment):
    id = 'market-maker-v0'

    def __init__(self, transaction_fee=LIMIT_ORDER_FEE, **kwargs):
        """
        Environment designed for automated market making.
        :param kwargs: refer to BaseEnvironment.py
        """
        super(MarketMaker, self).__init__(**kwargs)

        # environment attributes to override in sub-class
        self.actions = np.eye(17, dtype=np.float32)

        # get Broker class to keep track of PnL and orders
        self.broker = Broker(max_position=self.max_position,
                             transaction_fee=transaction_fee)

        self.action_space = spaces.Discrete(len(self.actions))
        self.reset()  # reset to load observation.shape
        self.observation_space = spaces.Box(low=-10, high=10,
                                            shape=self.observation.shape,
                                            dtype=np.float32)

        print('{} MarketMaker #{} instantiated\nobservation_space: {}'.format(
            self.sym, self._seed, self.observation_space.shape),
            'reward_type = {}'.format(self.reward_type))

    def __str__(self):
        return '{} | {}-{}'.format(MarketMaker.id, self.sym, self._seed)

    def map_action_to_broker(self, action: int):
        """
        Create or adjust orders per a specified action and adjust for penalties.
        :param action: (int) current step's action
        :return: (float) reward
        """
        reward = pnl = 0.0
        discouragement = 0.000000000001

        if action == 0:  # do nothing
            reward += discouragement

        elif action == 1:
            reward += self._create_order_at_level(reward, discouragement, level=0,
                                                  side='long')
            reward += self._create_order_at_level(reward, discouragement, level=4,
                                                  side='short')

        elif action == 2:
            reward += self._create_order_at_level(reward, discouragement, level=0,
                                                  side='long')
            reward += self._create_order_at_level(reward, discouragement, level=9,
                                                  side='short')
        elif action == 3:
            reward += self._create_order_at_level(reward, discouragement, level=0,
                                                  side='long')
            reward += self._create_order_at_level(reward, discouragement, level=14,
                                                  side='short')

        elif action == 4:
            reward += self._create_order_at_level(reward, discouragement, level=4,
                                                  side='long')
            reward += self._create_order_at_level(reward, discouragement, level=0,
                                                  side='short')

        elif action == 5:
            reward += self._create_order_at_level(reward, discouragement, level=4,
                                                  side='long')
            reward += self._create_order_at_level(reward, discouragement, level=4,
                                                  side='short')

        elif action == 6:

            reward += self._create_order_at_level(reward, discouragement, level=4,
                                                  side='long')
            reward += self._create_order_at_level(reward, discouragement, level=9,
                                                  side='short')
        elif action == 7:

            reward += self._create_order_at_level(reward, discouragement, level=4,
                                                  side='long')
            reward += self._create_order_at_level(reward, discouragement, level=14,
                                                  side='short')

        elif action == 8:
            reward += self._create_order_at_level(reward, discouragement, level=9,
                                                  side='long')
            reward += self._create_order_at_level(reward, discouragement, level=0,
                                                  side='short')

        elif action == 9:
            reward += self._create_order_at_level(reward, discouragement, level=9,
                                                  side='long')
            reward += self._create_order_at_level(reward, discouragement, level=4,
                                                  side='short')

        elif action == 10:
            reward += self._create_order_at_level(reward, discouragement, level=9,
                                                  side='long')
            reward += self._create_order_at_level(reward, discouragement, level=9,
                                                  side='short')

        elif action == 11:
            reward += self._create_order_at_level(reward, discouragement, level=9,
                                                  side='long')
            reward += self._create_order_at_level(reward, discouragement, level=14,
                                                  side='short')

        elif action == 12:
            reward += self._create_order_at_level(reward, discouragement, level=14,
                                                  side='long')
            reward += self._create_order_at_level(reward, discouragement, level=0,
                                                  side='short')

        elif action == 13:
            reward += self._create_order_at_level(reward, discouragement, level=14,
                                                  side='long')
            reward += self._create_order_at_level(reward, discouragement, level=4,
                                                  side='short')

        elif action == 14:
            reward += self._create_order_at_level(reward, discouragement, level=14,
                                                  side='long')
            reward += self._create_order_at_level(reward, discouragement, level=9,
                                                  side='short')

        elif action == 15:
            reward += self._create_order_at_level(reward, discouragement, level=14,
                                                  side='long')
            reward += self._create_order_at_level(reward, discouragement, level=14,
                                                  side='short')
        elif action == 16:
            reward += self.broker.flatten_inventory(self.best_bid, self.best_ask)
        else:
            print("L'action n'exist pas ! Il faut faire attention !!!")

        return reward, pnl

    def _create_position_features(self):
        """
        Create an array with features related to the agent's inventory
        :return: (np.array) normalized position features
        """
        return np.array((self.broker.long_inventory.position_count / self.max_position,
                         self.broker.short_inventory.position_count / self.max_position,
                         self.broker.get_total_pnl(self.best_bid, self.best_ask)
                         * self.broker.reward_scale,
                         self.broker.long_inventory.get_unrealized_pnl(self.best_bid)
                         * self.broker.reward_scale,
                         self.broker.short_inventory.get_unrealized_pnl(self.best_ask)
                         * self.broker.reward_scale,
                         self.broker.get_long_order_distance_to_midpoint(
                             midpoint=self.midpoint) * self.broker.reward_scale,
                         self.broker.get_short_order_distance_to_midpoint(
                             midpoint=self.midpoint) * self.broker.reward_scale,
                         *self.broker.get_queues_ahead_features()), dtype=np.float32)

    def _create_order_at_level(self, reward: float, discouragement: float, level=0,
                               side='long'):
        """
        Create a new order at a specified LOB level
        :param reward: (float) current step reward
        :param discouragement: (float) penalty deducted from reward for erroneous actions
        :param level: (int) level in the limit order book
        :param side: (str) direction of trade e.g., 'long' or 'short'
        :return: (float) reward with penalties added
        """
        adjustment = 1 if level > 0 else 0

        if side == 'long':
            best = self._get_book_data(MarketMaker.best_bid_index - level)
            denormalized_best = round(self.midpoint * (best + 1), 2)
            inside_best = self._get_book_data(
                MarketMaker.best_bid_index - level + adjustment)
            denormalized_inside_best = round(self.midpoint * (inside_best + 1), 2)
            plus_one = denormalized_best + 0.01

            if denormalized_inside_best == plus_one:
                # stick to best bid
                bid_price = denormalized_best
                # since LOB is rendered as cumulative notional, deduct the prior price
                # level to derive the notional value of orders ahead in the queue
                bid_queue_ahead = self._get_book_data(
                    MarketMaker.notional_bid_index - level) - self._get_book_data(
                    MarketMaker.notional_bid_index - level + adjustment)
            else:
                # insert a cent ahead to jump a queue
                bid_price = plus_one
                bid_queue_ahead = 0.

            bid_order = LimitOrder(ccy=self.sym, side='long', price=bid_price,
                                   step=self.local_step_number,
                                   queue_ahead=bid_queue_ahead)

            if self.broker.add(order=bid_order) is False:
                reward -= discouragement
            else:
                reward += discouragement

        if side == 'short':
            best = self._get_book_data(MarketMaker.best_ask_index + level)
            denormalized_best = round(self.midpoint * (best + 1), 2)
            inside_best = self._get_book_data(
                MarketMaker.best_ask_index + level - adjustment)
            denormalized_inside_best = round(self.midpoint * (inside_best + 1), 2)
            plus_one = denormalized_best + 0.01

            if denormalized_inside_best == plus_one:
                ask_price = denormalized_best
                # since LOB is rendered as cumulative notional, deduct the prior price
                # level to derive the notional value of orders ahead in the queue
                ask_queue_ahead = self._get_book_data(
                    MarketMaker.notional_ask_index + level) - self._get_book_data(
                    MarketMaker.notional_ask_index + level - adjustment)
            else:
                ask_price = plus_one
                ask_queue_ahead = 0.

            ask_order = LimitOrder(ccy=self.sym, side='short', price=ask_price,
                                   step=self.local_step_number,
                                   queue_ahead=ask_queue_ahead)

            if self.broker.add(order=ask_order) is False:
                reward -= discouragement
            else:
                reward += discouragement

        return reward
