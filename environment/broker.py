from environment.position import Position


class Broker(object):

    def __init__(self, max_position=1):
        self.max_position = max_position
        self.positions = dict()
        self.midpoints = dict()

    def add(self, order):
        if order.sym in self.positions:
            self.positions[order.sym].add(order=order)
        else:
            self.positions[order.sym] = Position(ccy=order.sym, max_position=self.max_position)
            self.positions[order.sym].add(order=order)

    def remove(self, order):
        if order.sym in self.positions:
            self.positions[order.sym].remove()
        else:
            print('Broker.remove() error with order = %s' % order)
            assert True is False

    def get_pnl(self, midpoint=100.0):
        pnl = 0.0
        for sym in self.positions.keys():
            pnl += self.positions[sym].get_unrealized_pnl(midpoint=midpoint)
        return pnl
