# order.py
#
#   Market and Limit order implementations for {broker/position}.py
#
#
from abc import ABC

from configurations import LIMIT_ORDER_FEE, LOGGER, MARKET_ORDER_FEE


class OrderMetrics(object):

    def __init__(self):
        """
        Class for capturing order / position metrics
        """
        self.drawdown_max = 0.0
        self.upside_max = 0.0
        self.steps_in_position = 0

    def __str__(self):
        return ('OrderMetrics: [ drawdown_max={} | upside_max={} | '
                'steps_in_position={} ]').format(self.drawdown_max, self.upside_max,
                                                 self.steps_in_position)


class Order(ABC):
    DEFAULT_SIZE = 1000.
    _id = 0
    LIMIT_ORDER_FEE = LIMIT_ORDER_FEE * 2
    MARKET_ORDER_FEE = MARKET_ORDER_FEE * 2

    def __init__(self, price: float, step: int, average_execution_price: float,
                 order_type='limit', ccy='BTC-USD', side='long', ):
        """

        :param price:
        :param step:
        :param average_execution_price:
        :param order_type:
        :param ccy:
        :param side:
        """
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
        Order._id += 1
        self.id = Order._id

    def __str__(self):
        return ' {} #{} | {} | {:.3f} | {} | {} | {}'.format(
            self.ccy, self.id, self.side, self.price, self.step, self.metrics,
            self.queue_ahead)

    @property
    def is_filled(self) -> bool:
        """
        If TRUE, the entire order has been executed.

        :return: (bool) TRUE if the order is completely filled
        """
        return self.executed >= Order.DEFAULT_SIZE

    def update_metrics(self, price: float, step: int) -> None:
        """
        Update specific position metrics per each order.

        :param price: (float) current midpoint price
        :param step: (int) current time step
        :return: (void)
        """
        self.metrics.steps_in_position = step - self.step
        if self.is_filled:
            if self.side == 'long':
                unrealized_pnl = (price - self.average_execution_price) / \
                                 self.average_execution_price
            elif self.side == 'short':
                unrealized_pnl = (self.average_execution_price - price) / \
                                 self.average_execution_price
            else:
                unrealized_pnl = 0.0
                LOGGER.warning('alert: unknown order.step() side %s' % self.side)

            if unrealized_pnl < self.metrics.drawdown_max:
                self.metrics.drawdown_max = unrealized_pnl

            if unrealized_pnl > self.metrics.upside_max:
                self.metrics.upside_max = unrealized_pnl


class MarketOrder(Order):
    def __init__(self, ccy='BTC-USD', side='long', price=0.0, step=-1):
        super(MarketOrder, self).__init__(price=price,
                                          step=step,
                                          average_execution_price=-1,
                                          order_type='market',
                                          ccy=ccy,
                                          side=side)

    def __str__(self):
        return "[MarketOrder] " + super(MarketOrder, self).__str__()


class LimitOrder(Order):

    def __init__(self, ccy='BTC-USD', side='long', price=0.0, step=-1, queue_ahead=100.):
        super(LimitOrder, self).__init__(price=price,
                                         step=step,
                                         average_execution_price=-1.,
                                         order_type='limit',
                                         ccy=ccy,
                                         side=side)
        self.queue_ahead = queue_ahead
        # print('LimitOrder_{}: [price={} | side={} | step={} | queue={}]'.format(
        #     self.ccy, self.price, self.side, self.step, self.queue_ahead
        # ))

    def __str__(self):
        return "[LimitOrder] " + super(LimitOrder, self).__str__()

    def reduce_queue_ahead(self, executed_volume=100.) -> None:
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

    def process_executions(self, volume=100.) -> None:
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

    def get_average_execution_price(self) -> float:
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
    def is_first_in_queue(self) -> bool:
        """
        Determine if current order is first in line to be executed.

        :return: True if the order is the first in the queue
        """
        return self.queue_ahead <= 0.
