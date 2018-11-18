import pandas as pd


class PositionI(object):

    def __init__(self, max_position=1):
        self.max_position_count = max_position
        self.positions = []
        self.position_count = 0
        self.position_exposure = 0.0
        self.position_avg_price = 0.0
        self.realized_pnl = 0.0

    def add(self, order):
        if self.position_count < self.max_position_count:
            self.positions.append(order)
            self._update(first_order_price=order['price'])
            print('PositionI.add() --> %s' % order)
        else:
            print('PositionI.add() have %i of %i positions open' % (self.position_count, self.max_position_count))

    def remove(self, order):
        if self.position_count > 0:
            first_order = self.positions.pop()  # FIFO order inventory
            self._update(first_order_price=-first_order['price'], order=order)
            print('PositionI.remove() --> %s' % first_order)
        else:
            print('PositionI.remove() never had positions')

    def reset(self):
        self.positions.clear()
        self.position_count = 0
        print('PositionI.reset() completed')

    def _update(self, first_order_price, order=None):
        self.position_exposure += first_order_price

        if order is not None:
            if order['side'] == 'long':
                realized_pnl = (order['price'] - first_order_price) / first_order_price * order.size
            elif order['side'] == 'short':
                realized_pnl = (first_order_price - order['price']) / first_order_price * order.size
            else:
                print('PositionI._update() Uknown order side for %s' % order)
            self.realized_pnl += realized_pnl

        self.position_count = len(self.positions)
        self.position_avg_price = self.position_exposure / self.position_count
        print('PositionI._update.Total positions: %i' % self.position_count)

    def get_unrealized_long_pnl(self, midpoint=100.0):
        pnl = 0.0
        for order in self.positions:
            pnl += order['price'] - midpoint
        return pnl

    def get_unrealized_short_pnl(self, midpoint=100.0):
        pnl = 0.0
        for order in self.positions:
            pnl += midpoint - order['price']
        return pnl


class Position(object):

    def __init__(self, ccy='BTC-USD', max_position=1):
        self.long_inventory = PositionI(max_position=max_position)
        self.short_inventory = PositionI(max_position=max_position)
        self.sym = ccy
        self.net_position_exposure = 0.0
        self.deal_number = 0
        self.transaction_details_columns = ['symbol', 'side', 'type', 'price', 'pnl', 'time', 'deal_number']
        self.transaction_details = pd.DataFrame(columns=self.transaction_details_columns)

    def add(self, order):
        if order['side'] == 'long':
            self.long_inventory.add(order=order)
        elif order['side'] == 'short':
            self.short_inventory.add(order=order)
        else:
            print('Position.add() unknown order.side = %s' % order)
            return

        self.net_position_exposure = self.long_inventory.position_exposure - self.short_inventory.position_exposure
        self._update_transaction_details(order=order)

    def remove(self, order):
        if order['side'] == 'long':
            self.long_inventory.remove()
        elif order['side'] == 'short':
            self.short_inventory.remove()
        else:
            print('Position.remove() unknown order.side = %s' % order['side'])
            return

        self.net_position_exposure = self.long_inventory.position_exposure - self.short_inventory.position_exposure
        self._update_transaction_details(order=order)

    def reset(self):
        self.long_inventory.reset()
        self.short_inventory.reset()
        self.net_position_exposure = 0.0
        self.deal_number = 0
        self.transaction_details = pd.DataFrame(columns=self.transaction_details_columns)

    def get_unrealized_pnl(self, midpoint=100.0):
        long_pnl = self.long_inventory.get_unrealized_long_pnl(midpoint=midpoint)
        short_pnl = self.short_inventory.get_unrealized_short_pnl(midpoint=midpoint)
        return long_pnl + short_pnl

    def get_realized_pnl(self):
        return self.short_inventory.realized_pnl + self.long_inventory.realized_pnl

    def _update_transaction_details(self, order):
        self.transaction_details.append(order)
        self.deal_number += 1
