# broker.py
#
#   Wrapper class implementing position.py to manage inventory and PnL
#
#
import logging
from gym_trading.utils.order import Order
from gym_trading.utils.position import Position

logging.basicConfig(level=logging.INFO, format='[%(asctime)s] %(message)s')
logger = logging.getLogger('broker')


class Broker(object):
    reward_scale = 0.01  # use as the denominator to scale PnL

    def __init__(self, max_position=1, transaction_fee=None):
        """
        Broker class is a wrapper for the PositionI class
        and is implemented in `gym_trading.py`
        :param max_position: (int) maximum number of positions agent can have open
                                    at a given time.
        :param transaction_fee: (float) fee to use for add/remove order transactions;
                If NONE, then transaction fees are omitted.
        """
        self.long_inventory = Position(side='long', max_position=max_position,
                                       transaction_fee=transaction_fee)
        self.short_inventory = Position(side='short', max_position=max_position,
                                        transaction_fee=transaction_fee)
        self.transaction_fee = transaction_fee

    def __str__(self):
        return self.long_inventory.__str__() + "\n" + self.short_inventory.__str__()

    def reset(self):
        """
        Reset long and short inventories
        :return: (void)
        """
        self.long_inventory.reset()
        self.short_inventory.reset()

    def add(self, order: Order):
        """
        Add / update an order
        :param order: (Order) New order to be used for updating existing order or
                        placing a new order
        :return: (bool) TRUE if order add action successfully completed, FALSE if already
                        at position_max or unknown order.side
        """
        if order.side == 'long':
            is_added = self.long_inventory.add(order=order)
        elif order.side == 'short':
            is_added = self.short_inventory.add(order=order)
        else:
            is_added = False
            logger.warning('Broker.add() unknown order.side = %s' % order)
        return is_added

    def remove(self, order: Order):
        """
        Remove position from inventory and return position PnL.
        :param order: (Order) Market order used to close position.
        :return: (bool) TRUE if position removed successfully
        """
        pnl = 0.
        if order.side == 'long':
            pnl += self.long_inventory.remove(price=order.price)
        elif order.side == 'short':
            pnl += self.short_inventory.remove(price=order.price)
        else:
            logger.warning('Broker.remove() unknown order.side = %s' % order.side)
        return pnl

    def flatten_inventory(self, bid_price: float, ask_price: float):
        """
        Flatten all positions held in inventory.
        :param bid_price: (float) current bid price
        :param ask_price: (float) current ask price
        :return: (float) PnL from flattening inventory
        """
        long_pnl = self.long_inventory.flatten_inventory(price=bid_price)
        short_pnl = self.short_inventory.flatten_inventory(price=ask_price)
        return long_pnl + short_pnl

    def get_unrealized_pnl(self, bid_price: float, ask_price: float):
        """
        Unrealized PnL as a percentage gain
        :return: (float) PnL %
        """
        long_pnl = self.long_inventory.get_unrealized_pnl(price=bid_price)
        short_pnl = self.short_inventory.get_unrealized_pnl(price=ask_price)
        return long_pnl + short_pnl

    @property
    def realized_pnl(self):
        """
        Realized PnL as a percentage gain
        :return: (float) PnL %
        """
        return self.short_inventory.realized_pnl + self.long_inventory.realized_pnl

    def get_total_pnl(self, bid_price: float, ask_price: float):
        """
        Unrealized + realized PnL.
        :param bid_price: (float) current bid price
        :param ask_price: (float) current ask price
        :return: (float) total PnL
        """
        return self.get_unrealized_pnl(bid_price, ask_price) + self.realized_pnl

    @property
    def long_inventory_count(self):
        """
        Number of long positions currently held in inventory.
        :return: (int) number of positions
        """
        return self.long_inventory.position_count

    @property
    def short_inventory_count(self):
        """
        Number of short positions currently held in inventory.
        :return: (int) number of positions
        """
        return self.short_inventory.position_count

    @property
    def total_trade_count(self):
        """
        Total number of long and short trades executed
        :return: (int) number of executed trades
        """
        return self.long_inventory.total_trade_count + \
            self.short_inventory.total_trade_count

    def step_limit_order_pnl(self, bid_price=100., ask_price=100., buy_volume=1000.,
                             sell_volume=1000., step=100):
        """
        Update PnL & positions every time step in the environment.
        :param bid_price: (float) current time step bid price
        :param ask_price: (float) current time step ask price
        :param buy_volume: (float) current time step buy volume
        :param sell_volume: (float) current time step sell volume
        :param step: (int) current time step number
        :return: (float) PnL for current time step due to limit order fill and netting
        """
        pnl = 0.
        if self.long_inventory.step(bid_price=bid_price, ask_price=ask_price,
                                    buy_volume=buy_volume, sell_volume=sell_volume,
                                    step=step):
            # check if we can net the inventory
            if self.short_inventory_count > 0:
                # net out the inventory
                new_position = self.long_inventory.pop_position()
                pnl += self.short_inventory.remove(
                    price=new_position.average_execution_price)
        if self.short_inventory.step(bid_price=bid_price, ask_price=ask_price,
                                     buy_volume=buy_volume, sell_volume=sell_volume,
                                     step=step):
            # check if we can net the inventory
            if self.long_inventory_count > 0:
                # net out the inventory
                new_position = self.short_inventory.pop_position()
                pnl += self.long_inventory.remove(
                    price=new_position.average_execution_price)
        return pnl

    def get_short_order_distance_to_midpoint(self, midpoint=100.):
        """
        Distance percentage between current midpoint price and the open order's
        posted price.
        :param midpoint: (float) current midpoint of crypto currency
        :return: (float) distance between open order and midpoint price
        """
        return self.short_inventory.get_distance_to_midpoint(midpoint=midpoint)

    def get_long_order_distance_to_midpoint(self, midpoint=100.):
        """
        Distance percentage between current midpoint price and the open order's
        posted price.
        :param midpoint: (float) current midpoint of crypto currency
        :return: (float) distance between open order and midpoint price
        """
        return self.long_inventory.get_distance_to_midpoint(midpoint=midpoint)

    def get_queues_ahead_features(self):
        """
        Scaled [-1, 1] ratio of how 'complete' an open order is. 'Complete' can be
        defined as a function with arguments:
            1. Open order's current rank in the LOB queue
            2. Partial fills in the open order
            3. Open order's quantity/size
        :return: (float) Scaled open order completion ratio [-1, 1]
        """
        buy_queue = short_queue = 0.

        if self.long_inventory.order:
            queue = self.long_inventory.order.queue_ahead
            executions = self.long_inventory.order.executed
            trade_size = self.long_inventory.order.DEFAULT_SIZE
            buy_queue = (executions - queue) / (queue + trade_size)

        if self.short_inventory.order:
            queue = self.short_inventory.order.queue_ahead
            executions = self.short_inventory.order.executed
            trade_size = self.short_inventory.order.DEFAULT_SIZE
            short_queue = (executions - queue) / (queue + trade_size)

        return buy_queue, short_queue
