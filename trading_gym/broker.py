import pandas as pd


class PositionI(object):

    def __init__(self, side='buy', max_position=1):
        self.max_position_count = max_position
        self.positions = []
        self.realized_pnl = 0.0
        self.unrealized_pnl = 0.0
        self.full_inventory = False
        self.position_count = 0
        self.total_exposure = 0.0
        self.side = side

    def add(self, order):
        if self.position_count < self.max_position_count:
            self.positions.append(order)
            self.position_count += 1
            self.total_exposure += order['price']
        # else:
        #     print('PositionI.add() have %i of %i positions open' % (self.position_count, self.max_position_count))

    def remove(self, order):
        if self.position_count > 0:
            opening_order = self.positions.pop()  # FIFO order inventory
            self.position_count -= 1
            self.total_exposure -= opening_order['price']
            if self.side == 'buy':
                self.realized_pnl += (order['price'] - opening_order['price']) / opening_order['price']
            elif self.side == 'sell':
                self.realized_pnl += (opening_order['price'] - order['price']) / opening_order['price']
            else:
                print('PositionI.remove() Warning - position side unrecognized = {}'.format(self.side))
        # else:
        #     print('PositionI.remove() no position in inventory to remove')

    def reset(self):
        self.positions.clear()
        self.realized_pnl = 0.0
        self.unrealized_pnl = 0.0
        self.full_inventory = False
        self.position_count = 0
        self.total_exposure = 0.0

    def get_unrealized_pnl(self, midpoint=100.):
        if self.position_count == 0:
            return 0.0

        average_price = self.total_exposure / self.position_count
        pnl = 0.0
        if self.side == 'buy':
            pnl = (midpoint / average_price) - 1.0
        elif self.side == 'sell':
            pnl = (average_price / midpoint) - 1.0
        else:
            print('PositionI.remove() Warning - position side unrecognized = {}'.format(self.side))

        return pnl


class Broker(object):

    def __init__(self, max_position=1):
        self.long_inventory = PositionI(side='buy', max_position=max_position)
        self.short_inventory = PositionI(side='sell', max_position=max_position)
        self.net_exposure = 0.0

    def add(self, order):
        if order['side'] == 'long':
            self.long_inventory.add(order=order)
        elif order['side'] == 'short':
            self.short_inventory.add(order=order)
        else:
            print('Broker.add() unknown order.side = %s' % order)
            return

        self.net_exposure = self.long_inventory.total_exposure - self.short_inventory.total_exposure

    def remove(self, order):
        if order['side'] == 'long':
            self.long_inventory.remove(order=order)
        elif order['side'] == 'short':
            self.short_inventory.remove(order=order)
        else:
            print('Broker.remove() unknown order.side = %s' % order['side'])
            return

        self.net_exposure = self.long_inventory.total_exposure - self.short_inventory.total_exposure

    def reset(self):
        self.long_inventory.reset()
        self.short_inventory.reset()

    def get_unrealized_pnl(self, midpoint=100.):
        long_pnl = self.long_inventory.get_unrealized_pnl(midpoint=midpoint)
        short_pnl = self.short_inventory.get_unrealized_pnl(midpoint=midpoint)
        return long_pnl + short_pnl

    def get_realized_pnl(self):
        return self.short_inventory.realized_pnl + self.long_inventory.realized_pnl
