
class PositionI(object):
    '''
    Position class keeps track the agent's trades
    and provides stats (e.g., pnl) on all trades
    '''

    def __init__(self, side='long', max_position=1):
        self.max_position_count = max_position
        self._positions = []
        self._realized_pnl = []
        self._steps_in_position = []
        self.unrealized_pnl = 0.0
        self.full_inventory = False
        self.position_count = 0
        self.total_exposure = 0.0
        self.side = side

    def add(self, order):
        if not self.full_inventory:
            self._positions.append(order)
            self.position_count += 1
            self.total_exposure += order['price']
            self.full_inventory = self.position_count >= self.max_position_count
            # print('  %s @ %.2f | step %i' % (order['side'], order['price'], order['step']))
            return True
        else:
            # print('  %s inventory max' % order['side'])
            return False

    def remove(self, order):
        if self.position_count > 0:
            opening_order = self._positions.pop()  # FIFO order inventory
            self.position_count -= 1
            self.total_exposure -= opening_order['price']
            self.full_inventory = self.position_count >= self.max_position_count
            self._steps_in_position.append(order['step'] - opening_order['step'])

            if self.side == 'long':
                self._realized_pnl.append((order['price'] - opening_order['price']) / opening_order['price'])
            elif self.side == 'short':
                self._realized_pnl.append((opening_order['price'] - order['price']) / opening_order['price'])
            else:
                print('PositionI.remove() Warning - position side unrecognized = {}'.format(self.side))

            return True
        else:
            return False

    def reset(self):
        self._positions.clear()
        self._steps_in_position.clear()
        self._realized_pnl = []
        self.unrealized_pnl = 0.0
        self.full_inventory = False
        self.position_count = 0
        self.total_exposure = 0.0

    @property
    def positions(self):
        return self._positions

    @property
    def realized_pnl(self):
        return self._realized_pnl

    @property
    def steps_in_position(self):
        return self._steps_in_position

    def get_unrealized_pnl(self, midpoint=100.):
        if self.position_count == 0:
            return 0.0

        average_price = self.total_exposure / self.position_count
        pnl = 0.0
        if self.side == 'long':
            pnl = (midpoint / average_price) - 1.0
            # pnl *= self.position_count
        elif self.side == 'short':
            pnl = (average_price / midpoint) - 1.0
            # pnl *= self.position_count
        else:
            print('PositionI.remove() Warning - position side unrecognized = {}'.format(self.side))

        return pnl

    def get_realized_pnl(self):  # TODO make sum() update within .remove() function call
        return sum(self._realized_pnl)

    def flatten_inventory(self, order):
        # print(' Flattening {} inventory: {} positions'.format(self.side, self.position_count))
        before_pnl = self.get_realized_pnl()
        [self.remove(order=order) for _ in range(self.position_count)]
        after_pnl = self.get_realized_pnl()
        return after_pnl - before_pnl


class Broker(object):
    '''
    Broker class is a wrapper for the PositionI class
    and is implemented in `trading_gym.py`
    '''
    def __init__(self, max_position=1):
        self.long_inventory = PositionI(side='long', max_position=max_position)
        self.short_inventory = PositionI(side='short', max_position=max_position)
        self.net_exposure = 0.0

    def add(self, order):
        ret = False
        if order['side'] == 'long':
            ret = self.long_inventory.add(order=order)
        elif order['side'] == 'short':
            ret = self.short_inventory.add(order=order)
        else:
            print('Broker.add() unknown order.side = %s' % order)

        self.net_exposure = self.long_inventory.total_exposure - self.short_inventory.total_exposure
        return ret

    def remove(self, order):
        ret = False
        if order['side'] == 'long':
            ret = self.long_inventory.remove(order=order)
        elif order['side'] == 'short':
            ret = self.short_inventory.remove(order=order)
        else:
            print('Broker.remove() unknown order.side = %s' % order['side'])

        self.net_exposure = self.long_inventory.total_exposure - self.short_inventory.total_exposure
        return ret

    def reset(self):
        self.long_inventory.reset()
        self.short_inventory.reset()

    def get_unrealized_pnl(self, midpoint=100.):
        long_pnl = self.long_inventory.get_unrealized_pnl(midpoint=midpoint)
        short_pnl = self.short_inventory.get_unrealized_pnl(midpoint=midpoint)
        return long_pnl + short_pnl

    def get_realized_pnl(self):
        return self.short_inventory.get_realized_pnl() + self.long_inventory.get_realized_pnl()

    def get_total_pnl(self, midpoint):
        total_pnl = self.get_unrealized_pnl(midpoint=midpoint)
        total_pnl += self.short_inventory.get_realized_pnl()
        total_pnl += self.long_inventory.get_realized_pnl()
        return total_pnl

    @property
    def long_inventory_count(self):
        return self.long_inventory.position_count

    @property
    def short_inventory_count(self):
        return self.short_inventory.position_count

    def flatten_inventory(self, order):
        long_pnl = self.long_inventory.flatten_inventory(order=order)
        short_pnl = self.short_inventory.flatten_inventory(order=order)
        return long_pnl + short_pnl

