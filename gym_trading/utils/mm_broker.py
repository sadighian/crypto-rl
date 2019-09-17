# mm_broker.py
#
#   Inventory and risk management for the MarketMaker environment
#
#
import logging
from configurations.configs import MARKET_ORDER_FEE, LIMIT_ORDER_FEE
import numpy as np


logging.basicConfig(level=logging.DEBUG, format='[%(asctime)s] %(message)s')
logger = logging.getLogger('mm_broker')


class Order(object):
    DEFAULT_SIZE = 1000.
    _id = 0

    def __init__(self, ccy='BTC-USD', side='long', price=0.0, step=-1, queue_ahead=100.):
        self.ccy = ccy
        self.side = side
        self.price = price
        self.step = step
        self.executed = 0.
        Order._id += 1
        self.id = Order._id
        self.queue_ahead = queue_ahead
        self.executions = dict()
        self.average_execution_price = -1.

    def __str__(self):
        return ' {}-{} | price={:.4f} | step={} | executed={:.2f} | queue={}'.format(
            self.ccy, self.side, self.price, self.step, self.executed, self.queue_ahead)

    def reduce_queue_ahead(self, executed_volume=100.):
        """
        Subtract transactions from the queue ahead of the agent's open order in the
        LOB. This attribute is used to inform the agent how much notional volume is
        ahead of it's open order.
        :param executed_volume: (float) notional volume of recent transaction
        :return: (void)
        """
        self.queue_ahead -= executed_volume
        if self.queue_ahead < 0.:
            splash = 0. - self.queue_ahead
            self.queue_ahead = 0.
            self.process_executions(volume=splash)

    def process_executions(self, volume=100.):
        """
        Subtract transactions from the agent's open order (e.g., partial fills).
        :param volume: (float) notional volume of recent transaction
        :return: (void)
        """
        self.executed += volume
        overflow = 0.
        if self.is_filled:
            overflow = self.executed - Order.DEFAULT_SIZE
            self.executed -= overflow

        _price = float(self.price)
        if _price in self.executions:
            self.executions[_price] += volume - overflow
        else:
            self.executions[_price] = volume - overflow

    def get_average_execution_price(self):
        """
        Average execution price of an order.

        Note: agents can update a given order many times, thus a single order can have
                partial fills at many different prices.
        :return: (float) average execution price
        """
        total_volume = sum([notional_volume / price for price, notional_volume in
                            self.executions.items()])

        self.average_execution_price = 0.
        for price, notional_volume in self.executions.items():
            self.average_execution_price += price * (
                    (notional_volume / price) / total_volume)

        self.average_execution_price = round(self.average_execution_price, 2)
        return self.average_execution_price

    @property
    def is_filled(self):
        return self.executed >= Order.DEFAULT_SIZE

    @property
    def is_first_in_queue(self):
        return self.queue_ahead <= 0.


class PositionI(object):
    """
    Position class keeps track the agent's trades
    and provides stats (e.g., pnl)
    """

    def __init__(self, side='long', max_position=1, include_fees=True):
        self.max_position_count = max_position
        self.positions = []
        self.order = None
        self.realized_pnl = 0.0
        self.unrealized_pnl = 0.0
        self.full_inventory = False
        self.total_exposure = 0.0
        self.side = side
        self.average_price = 0.0
        self.reward_size = 1 / self.max_position_count
        self.total_trade_count = 0
        self.include_fees = include_fees

    def __str__(self):
        msg = 'PositionI-{}: [realized_pnl={:.4f} | unrealized_pnl={:.4f}'.format(
            self.side, self.realized_pnl, self.unrealized_pnl)
        msg += ' | total_exposure={:.4f} | total_trade_count={}]'.format(
            self.total_exposure, self.total_trade_count)
        return msg

    def reset(self):
        self.positions.clear()
        self.order = None
        self.realized_pnl = 0.0
        self.unrealized_pnl = 0.0
        self.full_inventory = False
        self.total_exposure = 0.0
        self.average_price = 0.0
        self.total_trade_count = 0

    @property
    def position_count(self):
        return len(self.positions)

    def add_order(self, order: Order):
        if self.order is None:
            if self.full_inventory:
                logger.info(("{} order rejected. Already at max position limit "
                             "({})").format(self.side, self.max_position_count))
                return False
            self.order = order
            logger.debug('\nOpened new order={}'.format(order))

        elif self.order.price != order.price:
            self.order.price = order.price
            self.order.queue_ahead = order.queue_ahead
            self.order.id = order.id
            logger.debug('\nUpdating order{} --> \n{}'.format(order, self.order))
        else:
            logger.debug("\nNothing to update about the order {}".format(self.order))

        return True

    def cancel_order(self):
        if self.order is None:
            logger.debug('No {} open orders to cancel.'.format(self.side))
            return False
        logger.debug('Cancelling order ({})'.format(self.order))
        self.order = None
        return True

    def step(self, bid_price=100., ask_price=100., buy_volume=1000., sell_volume=1000.,
             step=100):
        if self.order is None:
            return False

        if self.order.side == 'long':
            if bid_price <= self.order.price:
                if self.order.is_first_in_queue:
                    self.order.process_executions(sell_volume)
                else:
                    self.order.reduce_queue_ahead(sell_volume)

        elif self.order.side == 'short':
            if ask_price >= self.order.price:
                if self.order.is_first_in_queue:
                    self.order.process_executions(buy_volume)
                else:
                    self.order.reduce_queue_ahead(buy_volume)

        if self.order.is_filled:
            avg_execution_px = self.order.get_average_execution_price()
            self.positions.append(self.order)
            self.total_exposure += avg_execution_px
            self.average_price = self.total_exposure / self.position_count
            self.full_inventory = self.position_count >= self.max_position_count
            steps_to_fill = step - self.order.step
            if self.include_fees:
                self.realized_pnl -= LIMIT_ORDER_FEE
            logger.debug('FILLED %s order #%i at %.3f after %i steps on %i.' % (
            self.order.side, self.order.id, avg_execution_px, steps_to_fill, step))
            # print('FILLED %s order #%i at %.3f after %i steps on %i.' % (
            #     self.order.side, self.order.id, avg_execution_px, steps_to_fill, step))
            self.order = None  # set the slot back to no open orders
            return True

        return False

    def pop_position(self):
        if self.position_count > 0:
            position = self.positions.pop()

            # update positions attributes
            self.total_exposure -= position.average_execution_price
            if self.position_count > 0:
                self.average_price = self.total_exposure / self.position_count
            else:
                self.average_price = 0

            self.full_inventory = self.position_count >= self.max_position_count
            logger.debug('---%s position #%i @ %.4f has been netted out.' % (
                self.side, position.id, position.price))
            return position
        else:
            logger.info('Error. No {} pop_position to remove.'.format(self.side))
            return None

    def remove_position(self, midpoint=100.):
        pnl = 0.
        if self.position_count > 0:
            order = self.positions.pop(0)
            # Calculate PnL
            if self.side == 'long':
                pnl = (midpoint - order.price) / order.price
            elif self.side == 'short':
                pnl = (order.price - midpoint) / order.price

            # Add Profit and Loss to total
            self.realized_pnl += pnl
            # update positions attributes
            self.total_exposure -= order.average_execution_price
            if self.position_count > 0:
                self.average_price = self.total_exposure / self.position_count
            else:
                self.average_price = 0
            self.full_inventory = self.position_count >= self.max_position_count
            self.total_trade_count += 1  # entry and exit = two trades
            print(
                'Netted {} position #{} with PnL={:.4f}'.format(self.side, order.id, pnl))
            return pnl
        else:
            logger.info('Error. No {} positions to remove.'.format(self.side))
            return pnl

    def flatten_inventory(self, midpoint=100.):
        logger.debug(
            '{} is flattening inventory of {}'.format(self.side, self.position_count))
        prev_realized_pnl = self.realized_pnl
        if self.position_count < 1:
            return -0.00000000001

        while self.position_count > 0:
            self.remove_position(midpoint=midpoint)
            if self.include_fees:
                self.realized_pnl -= MARKET_ORDER_FEE  # marker order fee

        return self.realized_pnl - prev_realized_pnl  # net change in PnL

    def get_unrealized_pnl(self, midpoint=100.):
        if self.position_count == 0:
            return 0.0

        difference = 0.0
        if self.side == 'long':
            difference = midpoint - self.average_price
        elif self.side == 'short':
            difference = self.average_price - midpoint

        if difference == 0.0:
            unrealized_pnl = 0.0
        else:
            unrealized_pnl = difference / self.average_price

        return unrealized_pnl

    def get_distance_to_midpoint(self, midpoint=100.):
        if self.order is None:
            return 0.
        if self.side == 'long':
            return (midpoint - self.order.price) / self.order.price
        elif self.side == 'short':
            return (self.order.price - midpoint) / self.order.price


class Broker(object):
    reward_scale = LIMIT_ORDER_FEE * 2.

    def __init__(self, max_position=1, include_fees=True):
        """
        Broker class is a wrapper for the PositionI class
        and is implemented in `gym_trading.py`
        :param max_position: (int) maximum number of positions agent can have open
                                    at a given time.
        :param include_fees: (bool) include transaction fees; if TRUE, then include
        """
        self.long_inventory = PositionI(side='long', max_position=max_position,
                                        include_fees=include_fees)
        self.short_inventory = PositionI(side='short', max_position=max_position,
                                         include_fees=include_fees)

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
            return self.long_inventory.add_order(order=order)
        elif order.side == 'short':
            return self.short_inventory.add_order(order=order)
        else:
            logger.warning(('Error. Broker trying to add to '
                            'the wrong side [{}]').format(order.side))
            return False

    def remove(self, side='long', midpoint=100.):
        """
        Remove position from inventory and return position PnL.
        :param side: (str) 'long' or 'short' direction of trade
        :param midpoint: (float) current midpoint price
        :return: (bool) TRUE if position remove action successfully completed, FALSE if
                        no positions are open or unknown order.side
        """
        if side == 'long':
            return self.long_inventory.remove_position(midpoint=midpoint)
        elif side == 'short':
            return self.short_inventory.remove_position(midpoint=midpoint)
        else:
            logger.warning(('Error. Broker trying to add to '
                            'the wrong side [{}]').format(side))
            return False

    def get_unrealized_pnl(self, midpoint=100.):
        """
        Unrealized PnL as a percentage gain
        :return: (float) PnL %
        """
        long_pnl = self.long_inventory.get_unrealized_pnl(midpoint=midpoint)
        short_pnl = self.short_inventory.get_unrealized_pnl(midpoint=midpoint)
        return long_pnl + short_pnl

    def get_realized_pnl(self):
        """
        Realized PnL as a percentage gain
        :return: (float) PnL %
        """
        return self.short_inventory.realized_pnl + self.long_inventory.realized_pnl

    def get_total_pnl(self, midpoint: float):
        """
        Unrealized + realized PnL.
        :param midpoint: (float) current midpoint price
        :return: (float) total PnL
        """
        total_pnl = self.get_unrealized_pnl(midpoint=midpoint) + self.get_realized_pnl()
        return total_pnl

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

    def flatten_inventory(self, bid_price=100., ask_price=100.):
        """
        Flatten all positions held in inventory.
        :param bid_price: (float) current bid price
        :param ask_price: (float) current ask price
        :return: (float) Scaled [0, ...) PnL from flattening inventory
        """
        total_pnl = self.long_inventory.flatten_inventory(midpoint=bid_price)
        total_pnl += self.short_inventory.flatten_inventory(midpoint=ask_price)
        if total_pnl != 0.:
            total_pnl /= Broker.reward_scale
        return total_pnl

    def step(self, bid_price=100., ask_price=100., buy_volume=1000., sell_volume=1000.,
             step=100):
        """
        Update PnL & positions every time step in the environment.
        :param bid_price: (float) current time step bid price
        :param ask_price: (float) current time step ask price
        :param buy_volume: (float) current time step buy volume
        :param sell_volume: (float) current time step sell volume
        :param step: (int) current time step number
        :return: (float) PnL for current time step
        """
        pnl = 0.

        if self.long_inventory.step(bid_price=bid_price, ask_price=ask_price,
                                    buy_volume=buy_volume, sell_volume=sell_volume,
                                    step=step):
            # check if we can net the inventory
            if self.short_inventory_count > 0:
                # net out the inventory
                new_position = self.long_inventory.pop_position()
                pnl += self.short_inventory.remove_position(
                    midpoint=new_position.price)  # if pnl != 0.:  #     pnl /=
                # Broker.reward_scale  #     logger.debug("step() pnl --> {
                # :.4f}".format(pnl))
        if self.short_inventory.step(bid_price=bid_price, ask_price=ask_price,
                                     buy_volume=buy_volume, sell_volume=sell_volume,
                                     step=step):
            # check if we can net the inventory
            if self.long_inventory_count > 0:
                # net out the inventory
                new_position = self.short_inventory.pop_position()
                pnl += self.long_inventory.remove_position(
                    midpoint=new_position.price)  # if pnl != 0.:  #     pnl /=
                # Broker.reward_scale  #     logger.debug("step() pnl --> {
                # :.4f}".format(pnl))
        scaled_pnl = pnl / Broker.reward_scale  # scale to [-1, 1] with 2x profit ratio
        return scaled_pnl

    def get_short_order_distance_to_midpoint(self, midpoint=100.):
        """
        Scaled [0, ...) distance between current midpoint price and the open order's
        posted price. The distance is scaled by the Broker's 'reward scale', which is a
        user input setting.
        :param midpoint: (float) current midpoint of crypto currency
        :return: (float) scaled distance between open order and midpoint price
        """
        return self.short_inventory.get_distance_to_midpoint(
            midpoint=midpoint) / Broker.reward_scale

    def get_long_order_distance_to_midpoint(self, midpoint=100.):
        """
        Scaled [0, ...) distance between current midpoint price and the open order's
        posted price. The distance is scaled by the Broker's 'reward scale', which is a
        user input setting.
        :param midpoint: (float) current midpoint of crypto currency
        :return: (float) scaled distance between open order and midpoint price
        """
        return self.long_inventory.get_distance_to_midpoint(
            midpoint=midpoint) / Broker.reward_scale

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

    def get_total_trade_count(self):
        """
        Total number of trades executed completely
        :return: (int) total trade count
        """
        return self.long_inventory.total_trade_count + \
               self.short_inventory.total_trade_count
