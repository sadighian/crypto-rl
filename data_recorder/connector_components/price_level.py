

class PriceLevel(object):

    def __init__(self, price=100.01, quantity=0.5):
        """
        PriceLevel constructor
        :param price: LOB adjust price level
        :param quantity: total quantity available at the price
        """
        # Core price level attributes
        self._price = price         # adjusted price level in LOB
        self._quantity = quantity   # total order size
        self._count = 0             # total number of orders
        self._notional = 0.         # total notional value of orders at price level
        # Trackers for order flow
        # Inspired by https://arxiv.org/abs/1907.06230v1
        self._limit_count = 0
        self._limit_quantity = 0.
        self._limit_notional = 0.
        self._market_count = 0
        self._market_quantity = 0.
        self._market_notional = 0.
        self._cancel_count = 0
        self._cancel_quantity = 0.
        self._cancel_notional = 0.

    def __str__(self):
        level_info = 'PriceLevel: [price={} | quantity={} | notional={}] \n'.format(
            self._price, self._quantity, self.notional)
        order_flow_info = ('_limit_count={} | _limit_quantity={} | _'
                           'market_count={} | ').format(
            self._limit_count, self._limit_quantity, self._market_count)
        order_flow_info += ('_market_quantity={} | _cancel_count={} | _'
                            'cancel_quantity={}').format(
            self._market_quantity, self._cancel_count, self._cancel_quantity)
        return level_info + order_flow_info

    @property
    def price(self):
        """
        Adjusted price of level in LOB
        :return: (float)
        """
        return self._price

    @property
    def quantity(self):
        """
        Total order size
        :return: (int)
        """
        return self._quantity

    @property
    def count(self):
        """
        Total number of orders
        :return: (int)
        """
        return self._count

    @property
    def notional(self):
        """
        Total notional value of the price level
        :return: (float)
        """
        return round(self._notional, 2)

    @property
    def limit_notional(self):
        """
        Total value of incoming limit orders added at the price level
        :return: (float)
        """
        return round(self._limit_notional, 2)

    @property
    def market_notional(self):
        """
        Total value of incoming market orders at the price level
        :return: (float)
        """
        return round(self._market_notional, 2)

    @property
    def cancel_notional(self):
        """
        Total value of incoming cancel orders at the price level
        :return: (float)
        """
        return round(self._cancel_notional, 2)

    def add_quantity(self, quantity=0.5, price=100.):
        """
        Add more orders to a given price level
        :param quantity: order size
        :param price: order price
        :return: (void)
        """
        self._quantity += quantity
        self._notional += quantity * price

    def remove_quantity(self, quantity=0.5, price=100.):
        """
        Remove more orders to a given price level
        :param quantity: order size
        :param price: order price
        :return: (void)
        """
        self._quantity -= quantity
        self._notional -= quantity * price

    def add_count(self):
        """
        Counter for number of orders received at price level
        :return: (void)
        """
        self._count += 1

    def remove_count(self):
        """
        Counter for number of orders received at price level
        :return: (void)
        """
        self._count -= 1

    def clear_trackers(self):
        """
        Reset all trackers back to zero at the start of a new LOB snapshot interval
        :return: (void)
        """
        self._limit_count = 0
        self._limit_quantity = 0.
        self._limit_notional = 0.
        self._market_count = 0
        self._market_quantity = 0.
        self._market_notional = 0.
        self._cancel_count = 0
        self._cancel_quantity = 0.
        self._cancel_notional = 0.

    def add_limit(self, quantity=0., price=100.):
        """
        Add new incoming limit order to trackers
        :param quantity: order size
        :param price: order price
        :return: (void)
        """
        self._limit_count += 1
        self._limit_quantity += quantity
        self._limit_notional += quantity * price

    def add_market(self, quantity=0., price=100.):
        """
        Add new incoming market order to trackers
        :param quantity: order size
        :param price: order price
        :return: (void)
        """
        self._market_count += 1
        self._market_quantity += quantity
        self._market_notional += quantity * price

    def add_cancel(self, quantity=0., price=100.):
        """
        Add new incoming cancel order to trackers
        :param quantity: order size
        :param price: order price
        :return: (void)
        """
        self._cancel_count += 1
        self._cancel_quantity += quantity
        self._cancel_notional += quantity * price
