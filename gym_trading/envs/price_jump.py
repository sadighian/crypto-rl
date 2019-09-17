from gym import spaces
from gym_trading.utils.broker import Broker, Order
from configurations.configs import INDICATOR_WINDOW_MAX
import logging
import numpy as np

from gym_trading.envs.base_env import BaseEnvironment


# logging
logging.basicConfig(level=logging.INFO, format='[%(asctime)s] %(message)s')
logger = logging.getLogger('PriceJump')


class PriceJump(BaseEnvironment):
    id = 'long-short-v0'

    def __init__(self, *,
                 fitting_file='LTC-USD_2019-04-07.csv.xz',
                 testing_file='LTC-USD_2019-04-08.csv.xz',
                 step_size=1,
                 max_position=5,
                 window_size=10,
                 seed=1,
                 action_repeats=10,
                 training=True,
                 format_3d=False,
                 z_score=True):
        super(PriceJump, self).__init__(
            fitting_file=fitting_file,
            testing_file=testing_file,
            step_size=step_size,
            max_position=max_position,
            window_size=window_size,
            seed=seed,
            action_repeats=action_repeats,
            training=training,
            format_3d=format_3d,
            z_score=z_score
        )

        self.action = 0
        # derive gym.env properties
        self.actions = np.eye(3, dtype=np.float32)

        # get Broker class to keep track of PnL and orders
        self.broker = Broker(max_position=max_position)

        self.action_space = spaces.Discrete(len(self.actions))
        self.reset()  # reset to load observation.shape
        self.observation_space = spaces.Box(low=-10,
                                            high=10,
                                            shape=self.observation.shape,
                                            dtype=np.float32)

        print('{} PriceJump #{} instantiated.\nself.observation_space.shape : {}'.format(
            self.sym, self._seed, self.observation_space.shape))

    def __str__(self):
        return '{} | {}-{}'.format(PriceJump.id, self.sym, self._seed)

    def step(self, action: int):
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
            # Pass current time step midpoint to broker to calculate PnL,
            # or if any open orders are to be filled
            buy_volume = self._get_book_data(PriceJump.buy_trade_index)
            sell_volume = self._get_book_data(PriceJump.sell_trade_index)

            self.tns.step(buys=buy_volume, sells=sell_volume)
            self.rsi.step(price=self.midpoint)

            self.broker.step(midpoint=self.midpoint)

            self.reward += self._send_to_broker(action=step_action)

            step_observation = self._get_step_observation(action=action)
            self.data_buffer.append(step_observation)

            if len(self.data_buffer) > self.window_size:
                del self.data_buffer[0]

            self.local_step_number += self.step_size

        self.observation = self._get_observation()

        if self.local_step_number > self.max_steps:
            self.done = True
            order = Order(ccy=self.sym, side=None, price=self.midpoint,
                          step=self.local_step_number)
            self.reward = self.broker.flatten_inventory(order=order)

        return self.observation, self.reward, self.done, {}

    def _send_to_broker(self, action: int):
        """
        Create or adjust orders per a specified action and adjust for penalties.
        :param action: (int) current step's action
        :return: (float) reward
        """
        reward = 0.0
        discouragement = 0.000000000001

        if action == 0:  # do nothing
            reward += discouragement

        elif action == 1:  # buy
            # price_fee_adjusted = self.midpoint + (PriceJump.fee * self.midpoint)
            if self.broker.short_inventory_count > 0:
                order = Order(ccy=self.sym, side='short',
                              price=self.midpoint,  #price_fee_adjusted,
                              step=self.local_step_number)
                self.broker.remove(order=order)
                reward += self.broker.get_reward(side=order.side) / \
                    self.broker.reward_scale  # scale realized PnL

            elif self.broker.long_inventory_count >= 0:
                order = Order(ccy=self.sym, side='long',
                              price=self.midpoint,  #price_fee_adjusted,
                              step=self.local_step_number)
                if self.broker.add(order=order) is False:
                    reward -= discouragement

            else:
                logger.info(('gym_trading.get_reward() ' +
                             'Error for action #{} - ' +
                             'unable to place an order with broker').format(action))

        elif action == 2:  # sell
            # price_fee_adjusted = self.midpoint - (PriceJump.fee * self.midpoint)
            if self.broker.long_inventory_count > 0:
                order = Order(ccy=self.sym, side='long',
                              price=self.midpoint,  #price_fee_adjusted,
                              step=self.local_step_number)
                self.broker.remove(order=order)
                reward += self.broker.get_reward(side=order.side) / \
                    self.broker.reward_scale  # scale realized PnL
            elif self.broker.short_inventory_count >= 0:
                order = Order(ccy=self.sym, side='short',
                              price=self.midpoint,  #price_fee_adjusted,
                              step=self.local_step_number)
                if self.broker.add(order=order) is False:
                    reward -= discouragement

            else:
                logger.info(('gym_trading.get_reward() ' +
                             'Error for action #{} - ' +
                             'unable to place an order with broker').format(action))

        else:
            logger.info(('Unknown action to take in get_reward(): ' +
                         'action={} | midpoint={}').format(action, self.midpoint))

        return reward

    def _create_position_features(self):
        """
        Create an array with features related to the agent's inventory
        :return: (np.array) normalized position features
        """
        return np.array(
            (self.broker.long_inventory.position_count / self.max_position,
             self.broker.short_inventory.position_count / self.max_position,
             self.broker.get_total_pnl(midpoint=self.midpoint) / PriceJump.target_pnl,
             self.broker.long_inventory.get_unrealized_pnl(self.midpoint) /
                self.broker.reward_scale,
             self.broker.short_inventory.get_unrealized_pnl(self.midpoint) /
                self.broker.reward_scale),
            dtype=np.float32)