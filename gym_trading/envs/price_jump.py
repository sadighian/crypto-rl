from gym import spaces
from configurations.configs import MARKET_ORDER_FEE
from gym_trading.envs.base_env import BaseEnvironment
from gym_trading.utils.broker import MarketOrder, Broker
import numpy as np


class PriceJump(BaseEnvironment):
    id = 'long-short-v0'

    def __init__(self, *, fitting_file='LTC-USD_2019-04-07.csv.xz',
                 testing_file='LTC-USD_2019-04-08.csv.xz', step_size=1, max_position=5,
                 window_size=10, seed=1, action_repeats=10, training=True,
                 format_3d=False, z_score=True, reward_type='trade_completion',
                 scale_rewards=True, alpha=None):
        super(PriceJump, self).__init__(fitting_file=fitting_file,
                                        testing_file=testing_file, step_size=step_size,
                                        max_position=max_position,
                                        window_size=window_size, seed=seed,
                                        action_repeats=action_repeats, training=training,
                                        format_3d=format_3d, z_score=z_score,
                                        reward_type=reward_type,
                                        scale_rewards=scale_rewards, alpha=alpha)

        self.actions = np.eye(3, dtype=np.float32)

        # get Broker class to keep track of PnL and orders
        self.broker = Broker(max_position=max_position, transaction_fee=MARKET_ORDER_FEE)

        self.action_space = spaces.Discrete(len(self.actions))
        self.reset()  # reset to load observation.shape
        self.observation_space = spaces.Box(low=-10, high=10,
                                            shape=self.observation.shape,
                                            dtype=np.float32)

        print('{} PriceJump #{} instantiated.\nself.observation_space.shape : {}'.format(
            self.sym, self._seed, self.observation_space.shape))

    def __str__(self):
        return '{} | {}-{}'.format(PriceJump.id, self.sym, self._seed)

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

        elif action == 1:  # buy
            if self.broker.short_inventory_count > 0:
                order = MarketOrder(ccy=self.sym, side='short', price=self.midpoint,
                                    step=self.local_step_number)
                pnl += self.broker.remove(order=order)

            elif self.broker.long_inventory_count >= 0:
                order = MarketOrder(ccy=self.sym, side='long', price=self.midpoint,
                                    # price_fee_adjusted,
                                    step=self.local_step_number)
                if self.broker.add(order=order) is False:
                    reward -= discouragement

            else:
                print(('gym_trading.get_reward() ' + 'Error for action #{} - ' +
                       'unable to place an order with broker').format(action))

        elif action == 2:  # sell
            if self.broker.long_inventory_count > 0:
                order = MarketOrder(ccy=self.sym, side='long', price=self.midpoint,
                                    step=self.local_step_number)
                pnl += self.broker.remove(order=order)

            elif self.broker.short_inventory_count >= 0:
                order = MarketOrder(ccy=self.sym, side='short', price=self.midpoint,
                                    step=self.local_step_number)
                if self.broker.add(order=order) is False:
                    reward -= discouragement

            else:
                print(('gym_trading.get_reward() ' + 'Error for action #{} - ' +
                            'unable to place an order with broker').format(action))

        else:
            print(('Unknown action to take in get_reward(): ' +
                              'action={} | midpoint={}').format(action, self.midpoint))

        return reward, pnl

    def _create_position_features(self):
        """
        Create an array with features related to the agent's inventory
        :return: (np.array) normalized position features
        """
        return np.array((self.broker.long_inventory.position_count / self.max_position,
                         self.broker.short_inventory.position_count / self.max_position,
                         self.broker.realized_pnl / self.broker.reward_scale,
                         self.broker.get_unrealized_pnl(self.best_bid, self.best_ask)
                         / self.broker.reward_scale),
                        dtype=np.float32)
