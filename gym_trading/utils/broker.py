# broker.py
#
#   Inventory and risk management module for environments
#
#
from abc import ABC
import logging
from configurations.configs import MARKET_ORDER_FEE, LIMIT_ORDER_FEE

logging.basicConfig(level=logging.INFO, format='[%(asctime)s] %(message)s')
logger = logging.getLogger('broker')


class OrderMetrics(object):

    def __init__(self):
        self.drawdown_max = 0.0
        self.upside_max = 0.0
        self.steps_in_position = 0

    def __str__(self):
        return ('OrderMetrics: [ drawdown_max={} | upside_max={} | '
                'steps_in_position={} ]').format(self.drawdown_max, self.upside_max,
                                                 self.steps_in_position)


class Order(ABC):
    DEFAULT_SIZE = 1000.
    id = 0
    LIMIT_ORDER_FEE = LIMIT_ORDER_FEE * 2
    MARKET_ORDER_FEE = MARKET_ORDER_FEE * 2

    def __init__(self, price: float, step: int, average_execution_price: float,
                 order_type='limit', ccy='BTC-USD', side='long', ):
        self.order_type = order_type
        self.ccy = ccy
        self.side = side
        self.price = price
        self.step = step
        self.average_execution_price = average_execution_price
        self.metrics = OrderMetrics()
        self.executed = 0.
        self.queue_ahead = 0.
        self.executions = dict()
        Order.id += 1

    def __str__(self):
        return ' {} | {} | {:.3f} | {} | {} | {}'.format(
            self.ccy, self.side, self.price, self.step, self.metrics, self.queue_ahead)

    @property
    def is_filled(self):
        return self.executed >= Order.DEFAULT_SIZE

    def update_metrics(self, price: float, step: int):
        """
        Update specific position metrics per each order.
        :param price: (float) current midpoint price
        :param step: (int) current time step
        :return: (void)
        """
        self.metrics.steps_in_position = step - self.step
        if self.is_filled:
            if self.side == 'long':
                unrealized_pnl = (
                                         price - self.average_execution_price) / \
                                 self.average_execution_price
            elif self.side == 'short':
                unrealized_pnl = (
                                         self.average_execution_price - price) / \
                                 self.average_execution_price
            else:
                unrealized_pnl = 0.0
                logger.warning('alert: unknown order.step() side %s' % self.side)

            if unrealized_pnl < self.metrics.drawdown_max:
                self.metrics.drawdown_max = unrealized_pnl

            if unrealized_pnl > self.metrics.upside_max:
                self.metrics.upside_max = unrealized_pnl


class MarketOrder(Order):
    def __init__(self, ccy='BTC-USD', side='long', price=0.0, step=-1):
        super(MarketOrder, self).__init__(price=price, step=step,
                                          average_execution_price=-1, order_type='market',
                                          ccy=ccy,
                                          side=side)

    def __str__(self):
        return "[MarketOrder] " + super(MarketOrder, self).__str__()


class LimitOrder(Order):

    def __init__(self, ccy='BTC-USD', side='long', price=0.0, step=-1, queue_ahead=100.):
        super(LimitOrder, self).__init__(price=price, step=step,
                                         average_execution_price=-1., order_type='limit',
                                         ccy=ccy, side=side)
        self.queue_ahead = queue_ahead

    def __str__(self):
        return "[LimitOrder] " + super(LimitOrder, self).__str__()

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
        # print(
        #     "\nprocess_executions: _price={} | executed={}".format(_price, self.executed))
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
        self.average_execution_price = sum(
            [notional_volume * price for price, notional_volume in
             self.executions.items()]) / self.DEFAULT_SIZE
        return round(self.average_execution_price, 2)

    @property
    def is_first_in_queue(self):
        """

        :return:
        """
        return self.queue_ahead <= 0.


class Position(ABC):
    slippage = MARKET_ORDER_FEE / 4

    def __init__(self, side='long', max_position=1, transaction_fee=None):
        """
        Position class keeps track the agent's trades and provides stats (e.g.,
        pnl) on all trades.
        :param side: 'long' or 'short'
        :param max_position: (int) maximum number of positions agent can have open
                                    at a given time.
        :param transaction_fee: (float) fee to use for add/remove order transactions;
                If NONE, then transaction fees are omitted.
        """
        self.max_position_count = max_position
        self.positions = []
        self.realized_pnl = 0.0
        self.full_inventory = False
        self.total_exposure = 0.0
        self.side = side
        self.average_price = 0.0
        self.total_trade_count = 0
        self.transaction_fee = transaction_fee
        self.order = None

    def __str__(self):
        msg = 'PositionI-{}: [realized_pnl={:.4f}'.format(self.side, self.realized_pnl)
        msg += ' | total_exposure={:.4f} | total_trade_count={}]'.format(
            self.total_exposure, self.total_trade_count)
        return msg

    def reset(self):
        """

        :return:
        """
        self.positions.clear()
        self.realized_pnl = 0.0
        self.full_inventory = False
        self.total_exposure = 0.0
        self.average_price = 0.0
        self.total_trade_count = 0
        self.order = None

    @property
    def position_count(self):
        """

        :return:
        """
        return len(self.positions)

    def _process_transaction_volume(self, volume: float):
        """

        :param volume:
        :return:
        """
        if self.order.is_first_in_queue:
            self.order.process_executions(volume)
        else:
            self.order.reduce_queue_ahead(volume)

    def _step_limit_order(self, bid_price: float, ask_price: float, buy_volume: float,
                          sell_volume: float, step: int):
        """

        :param bid_price:
        :param ask_price:
        :param buy_volume:
        :param sell_volume:
        :param step:
        :return:
        """
        if self.order is None:
            return False

        if self.order.side == 'long':
            if bid_price <= self.order.price:
                self._process_transaction_volume(volume=sell_volume)

        elif self.order.side == 'short':
            if ask_price >= self.order.price:
                self._process_transaction_volume(volume=buy_volume)

        if self.order.is_filled:
            avg_execution_px = self.order.get_average_execution_price()
            self.positions.append(self.order)
            self.total_exposure += avg_execution_px
            self.average_price = self.total_exposure / self.position_count
            self.full_inventory = self.position_count >= self.max_position_count
            self.total_trade_count += 1
            print('FILLED {} order #{} at {:.3f} after {} steps on {}.'.format(
                self.order.side, self.order.id, avg_execution_px,
                self.order.metrics.steps_in_position, step))
            self.order = None  # set the slot back to no open orders
            return True

        return False

    def _step_position_metrics(self, bid_price: float, ask_price: float, step: int):
        """

        :param bid_price:
        :param ask_price:
        :param step:
        :return:
        """
        price = bid_price if self.side == 'long' else ask_price
        if self.order:
            self.order.update_metrics(price=price, step=step)
        if self.position_count > 0:
            for position in self.positions:
                position.update_metrics(price=price, step=step)

    def step(self, bid_price: float, ask_price: float, buy_volume: float,
             sell_volume: float, step: int):
        """

        :param bid_price:
        :param ask_price:
        :param buy_volume:
        :param sell_volume:
        :param step:
        :return:
        """
        is_filled = self._step_limit_order(bid_price=bid_price, ask_price=ask_price,
                                           buy_volume=buy_volume, sell_volume=sell_volume,
                                           step=step)
        self._step_position_metrics(bid_price=bid_price, ask_price=ask_price, step=step)
        return is_filled

    def cancel_limit_order(self):
        """

        :return:
        """
        if self.order is None:
            logger.debug('No {} open orders to cancel.'.format(self.side))
            return False

        logger.debug('Cancelling order ({})'.format(self.order))
        self.order = None
        return True

    def _add_market_order(self, order: Order):
        """

        :param order:
        :return:
        """
        if self.full_inventory:
            logger.debug('  %s inventory max' % order.side)
            return False

        # Create a hypothetical average execution price incorporating a fixed slippage
        hypothetical_avg_price = order.price + Position.slippage \
            if order.side == 'long' else order.price - Position.slippage
        order.average_execution_price = hypothetical_avg_price
        order.executed = Order.DEFAULT_SIZE

        # Update position inventory attributes
        self.cancel_limit_order()  # remove any unfilled limit orders
        self.positions.append(order)  # execute and save the market order
        self.total_exposure += order.average_execution_price
        self.average_price = self.total_exposure / self.position_count
        self.full_inventory = self.position_count >= self.max_position_count
        self.total_trade_count += 1
        logger.debug('  %s @ %.2f | step %i' % (
            order.side, order.average_execution_price, order.step))
        return True

    def _add_limit_order(self, order: Order):
        """

        :param order:
        :return:
        """
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

    def add(self, order: Order):
        if order.order_type == 'market':
            return self._add_market_order(order=order)
        elif order.order_type == 'limit':
            return self._add_limit_order(order=order)
        else:
            logger.warning(
                "Position() add --> unknown order_type {}".format(order.order_type))

    def remove(self, price: float):
        """

        :param price:
        :return:
        """
        pnl = 0.
        if self.position_count < 1:
            logger.info('Error. No {} positions to remove.'.format(self.side))
            return pnl

        order = self.positions.pop(0)

        # Calculate PnL
        if self.side == 'long':
            pnl = (price - order.average_execution_price) / order.average_execution_price
        elif self.side == 'short':
            pnl = (order.average_execution_price - price) / order.average_execution_price

        # Add transaction fees, if provided
        if self.transaction_fee:
            pnl -= self.transaction_fee * 2  # round trip fees for position

        # Add Profit and Loss to realized gains/losses
        self.realized_pnl += pnl

        # Update positions attributes
        self.total_exposure -= order.average_execution_price
        self.average_price = self.total_exposure / self.position_count if \
            self.position_count > 0 else 0.
        self.full_inventory = self.position_count >= self.max_position_count
        logger.info(
            'Netted {} position #{} with PnL={:.4f}'.format(self.side, order.id, pnl))
        return pnl

    def pop_position(self):
        """

        :return:
        """
        if self.position_count > 0:
            position = self.positions.pop(0)

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

    def get_unrealized_pnl(self, price: float):
        """

        :param price:
        :return:
        """
        if self.position_count == 0:
            return 0.0

        if self.side == 'long':
            unrealized_pnl = (price - self.average_price) / self.average_price
        elif self.side == 'short':
            unrealized_pnl = (self.average_price - price) / self.average_price
        else:
            unrealized_pnl = 0.0
            logger.info(('Error: PositionI.get_unrealized_pnl() for '
                         'side = {}').format(self.side))

        # Add transaction fees, if provided
        if self.transaction_fee:
            unrealized_pnl -= self.transaction_fee * 2  # include round trip fees

        return unrealized_pnl

    def flatten_inventory(self, price: float):
        """

        :param price:
        :return:
        """
        logger.debug(
            '{} is flattening inventory of {}'.format(self.side, self.position_count))

        if self.position_count < 1:
            return -0.00000000001

        pnl = 0.
        while self.position_count > 0:
            pnl += self.remove(price=price)

        return pnl

    def get_distance_to_midpoint(self, midpoint: float):
        """

        :param midpoint:
        :return:
        """
        if self.order is None:
            return 0.
        return abs(midpoint - self.order.price) / self.order.price


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
        Scaled [0, ...) distance between current midpoint price and the open order's
        posted price. The distance is scaled by the Broker's 'reward scale', which is a
        user input setting.
        :param midpoint: (float) current midpoint of crypto currency
        :return: (float) distance between open order and midpoint price
        """
        return self.short_inventory.get_distance_to_midpoint(midpoint=midpoint)

    def get_long_order_distance_to_midpoint(self, midpoint=100.):
        """
        Scaled [0, ...) distance between current midpoint price and the open order's
        posted price. The distance is scaled by the Broker's 'reward scale', which is a
        user input setting.
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
