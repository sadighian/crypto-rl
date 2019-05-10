# Inventory and risk management for the MarketMaker environment
import logging
import numpy as np


logging.basicConfig(level=logging.INFO, format='[%(asctime)s] %(message)s')
logger = logging.getLogger('broker2')


class Order(object):
    _size = 1000.
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

    def __str__(self):
        return ' %s-%s | %.3f | %i | %.2f | %.2f' % \
               (self.ccy, self.side, self.price, self.step, self.executed, self.queue_ahead)

    @property
    def is_filled(self):
        return self.executed >= Order._size

    @property
    def is_first_in_queue(self):
        return self.queue_ahead <= 0.


class PositionI(object):
    """
    Position class keeps track the agent's trades
    and provides stats (e.g., pnl)
    """
    # TODO Add net position to calculate pnl
    def __init__(self, side='long', max_position=1):
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

    def reset(self):
        self.positions.clear()
        self.order = None
        self.realized_pnl = 0.0
        self.unrealized_pnl = 0.0
        self.full_inventory = False
        self.total_exposure = 0.0
        self.average_price = 0.0

    @property
    def position_count(self):
        return len(self.positions)

    def add_order(self, order):
        if not self.full_inventory:
            if self.order is None:
                logger.debug('Opened new order={}'.format(order))
            else:
                logger.debug('Updating existing order{} --> {}'.format(self.order, order))
            self.order = order
            return True
        else:
            logger.debug("{} order rejected. Already at max position limit ({})".format(
                self.side, self.max_position_count))
            return False

    def cancel_order(self):
        if self.order is None:
            logger.debug('No {} open orders to cancel.'.format(self.side))
            return False
        logger.debug('Cancelling order ({})'.format(self.order))
        self.order = None
        return True

    def step(self, bid_price=100., ask_price=100., buy_volume=1000., sell_volume=1000., step=100):
        if self.order is None:
            return False

        if self.order.side == 'long':
            if bid_price <= self.order.price:
                if self.order.is_first_in_queue:
                    self.order.executed += buy_volume
                else:
                    self.order.queue_ahead -= buy_volume
        elif self.order.side == 'short':
            if ask_price >= self.order.price:
                if self.order.is_first_in_queue:
                    self.order.executed += sell_volume
                else:
                    self.order.queue_ahead -= sell_volume

        if self.order.is_filled:
            self.positions.append(self.order)
            self.total_exposure += self.order.price
            self.average_price = self.total_exposure / self.position_count
            self.full_inventory = self.position_count >= self.max_position_count
            steps_to_fill = step - self.order.step
            logger.info('FILLED %s order #%i at %.3f after %i steps on %i.' %
                        (self.side, self.order.id, self.order.price, steps_to_fill, step))
            self.order = None  # set the slot back to no open orders
            return True

        return False

    def pop_position(self):
        if self.position_count > 0:
            position = self.positions.pop(0)

            # update positions attributes
            self.total_exposure -= position.price
            if self.total_exposure > 0:
                self.average_price = self.total_exposure / self.position_count
            else:
                self.average_price = 0

            self.full_inventory = self.position_count >= self.max_position_count
            logger.debug('---%s position #%i has been netted out.' % (self.side, position.id))
            return position
        else:
            logger.warning('Error. No {} pop_position to remove.'.format(self.side))
            return None

    def remove_position(self, midpoint=100.):
        pnl = 0.
        if self.position_count > 0:
            order = self.positions.pop()
            # Calculate PnL
            if self.side == 'long':
                pnl = (midpoint - order.price) / order.price
            elif self.side == 'short':
                pnl = (order.price - midpoint) / order.price
            # Add Profit and Loss to total
            self.realized_pnl += pnl
            # update positions attributes
            self.total_exposure -= order.price
            if self.total_exposure > 0:
                self.average_price = self.total_exposure / self.position_count
            else:
                self.average_price = 0
            self.full_inventory = self.position_count >= self.max_position_count
            logger.info('Removing %s position #%i. PnL=%.4f\n' %
                        (self.side, order.id, pnl))
            return True
        else:
            logger.warning('Error. No {} positions to remove.'.format(self.side))
            return False

    def flatten_inventory(self, midpoint=100.):
        prev_realized_pnl = self.realized_pnl
        logger.debug('{} is flattening inventory of {}'.format(self.side, self.position_count))
        while self.position_count > 0:
            self.remove_position(midpoint=midpoint)

        return self.realized_pnl - prev_realized_pnl

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
    '''
    Broker class is a wrapper for the PositionI class
    and is implemented in `gym_trading.py`
    '''
    def __init__(self, max_position=1):
        self.long_inventory = PositionI(side='long', max_position=max_position)
        self.short_inventory = PositionI(side='short', max_position=max_position)

    def reset(self):
        self.long_inventory.reset()
        self.short_inventory.reset()

    def add(self, order):
        if order.side == 'long':
            return self.long_inventory.add_order(order=order)
        elif order.side == 'short':
            return self.short_inventory.add_order(order=order)
        else:
            logger.warning('Error. Broker trying to add to the wrong side [{}]'.format(
                order.side))
            return False

    def get_unrealized_pnl(self, midpoint=100.):
        long_pnl = self.long_inventory.get_unrealized_pnl(midpoint=midpoint)
        short_pnl = self.short_inventory.get_unrealized_pnl(midpoint=midpoint)
        return long_pnl + short_pnl

    def get_realized_pnl(self):
        return self.short_inventory.realized_pnl + self.long_inventory.realized_pnl

    def get_total_pnl(self, midpoint):
        total_pnl = self.get_unrealized_pnl(midpoint=midpoint) + self.get_realized_pnl()
        return total_pnl

    @property
    def long_inventory_count(self):
        return self.long_inventory.position_count

    @property
    def short_inventory_count(self):
        return self.short_inventory.position_count

    def flatten_inventory(self, bid_price=100., ask_price=100.):
        long_pnl = self.long_inventory.flatten_inventory(midpoint=bid_price)
        short_pnl = self.short_inventory.flatten_inventory(midpoint=ask_price)
        return long_pnl + short_pnl

    def step(self,  bid_price=100., ask_price=100., buy_volume=1000., sell_volume=1000., step=100):

        if self.long_inventory.step(bid_price=bid_price, ask_price=ask_price,
                                    buy_volume=buy_volume, sell_volume=sell_volume, step=step):
            # check if we can net the inventory
            if self.short_inventory_count > 0:
                # net out the inventory
                new_position = self.long_inventory.pop_position()
                self.short_inventory.remove_position(midpoint=new_position.price)

        if self.short_inventory.step(bid_price=bid_price, ask_price=ask_price,
                                     buy_volume=buy_volume, sell_volume=sell_volume, step=step):
            # check if we can net the inventory
            if self.long_inventory_count > 0:
                # net out the inventory
                new_position = self.short_inventory.pop_position()
                self.long_inventory.remove_position(midpoint=new_position.price)

    def get_short_order_distance_to_midpoint(self, midpoint=100.):
        return self.short_inventory.get_distance_to_midpoint(midpoint=midpoint)

    def get_long_order_distance_to_midpoint(self, midpoint=100.):
        return self.long_inventory.get_distance_to_midpoint(midpoint=midpoint)
