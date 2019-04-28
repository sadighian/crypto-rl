

class Order:

    def __init__(self, ccy='BTC-USD', side='long', price=0.0, step=-1):
        self.ccy = ccy
        self.side = side
        self.price = price
        self.step = step
        self.drawdown_max = 0.0
        self.upside_max = 0.0

    def __str__(self):
        return ' %s | %s | %.3f | %i | %.3f | %.3f' % \
               (self.ccy, self.side, self.price, self.step, self.drawdown_max, self.upside_max)

    def update(self, midpoint=100.0):

        if self.side == 'long':
            unrealized_pnl = (midpoint - self.price) / self.price
        elif self.side == 'short':
            unrealized_pnl = (self.price - midpoint) / self.price
        else:
            unrealized_pnl = 0.0
            print('alert: uknown order.step() side %s' % self.side)

        if unrealized_pnl < self.drawdown_max:
            self.drawdown_max = unrealized_pnl

        if unrealized_pnl > self.upside_max:
            self.upside_max = unrealized_pnl


class PositionI(object):
    '''
    Position class keeps track the agent's trades
    and provides stats (e.g., pnl) on all tradself.average_price = es
    '''

    def __init__(self, side='long', max_position=1):
        self.max_position_count = max_position
        self.positions = []
        self.realized_pnl = 0.0
        self.unrealized_pnl = 0.0
        self.full_inventory = False
        self.position_count = 0
        self.total_exposure = 0.0
        self.side = side
        self.average_price = 0.0
        self.last_trade = {
            'steps_in_position': 0,
            'upside_max': 0.0,
            'drawdown_max': 0.0,
            'realized_pnl': 0.0
        }

    def reset(self):
        self.positions.clear()
        self.position_count = 0
        self.realized_pnl = 0.0
        self.unrealized_pnl = 0.0
        self.full_inventory = False
        self.total_exposure = 0.0
        self.average_price = 0.0
        # self.last_trade = {}

    def step(self, midpoint=100.0):
        if self.position_count > 0:
            for position in self.positions:
                position.update(midpoint=midpoint)

    def add(self, order):
        if not self.full_inventory:
            self.positions.append(order)
            self.position_count += 1
            self.total_exposure += order.price
            self.average_price = self.total_exposure / self.position_count
            self.full_inventory = self.position_count >= self.max_position_count
            # print('  %s @ %.2f | step %i' % (order.side, order.price, order.step))
            return True
        else:
            # print('  %s inventory max' % order['side'])
            return False

    def remove(self, order):
        if self.position_count > 0:
            opening_order = self.positions.pop()  # FIFO order inventory
            self.position_count -= 1
            self.total_exposure -= opening_order.price

            if self.position_count == 0:
                self.average_price = 0.0
            else:
                self.average_price = self.total_exposure / self.position_count

            self.full_inventory = self.position_count >= self.max_position_count
            self.last_trade['steps_in_position'] = order.step - opening_order.step
            self.last_trade['upside_max'] = opening_order.upside_max
            self.last_trade['drawdown_max'] = opening_order.drawdown_max

            if self.side == 'long':
                realized_trade_pnl = (order.price - opening_order.price) / opening_order.price
                self.realized_pnl += realized_trade_pnl
            elif self.side == 'short':
                realized_trade_pnl = (opening_order.price - order.price) / opening_order.price
                self.realized_pnl += realized_trade_pnl
            else:
                realized_trade_pnl = 0.0
                print('PositionI.remove() Warning - position side unrecognized = {}'.format(self.side))

            self.last_trade['realized_pnl'] = realized_trade_pnl

            # if realized_trade_pnl > 0.0:
            #     print(' %s PNL %.3f | upside %.3f / downside %.3f | steps: %i' % (self.side, realized_trade_pnl,
            #                                                self.last_trade['upside_max'],
            #                                                self.last_trade['drawdown_max'],
            #                                                self.last_trade['steps_in_position']))
            return True
        else:
            return False

    def get_unrealized_pnl(self, midpoint=100.):
        if self.position_count == 0:
            return 0.0

        if self.side == 'long':
            unrealized_pnl = (midpoint - self.average_price) / self.average_price
        elif self.side == 'short':
            unrealized_pnl = (self.average_price - midpoint) / self.average_price
        else:
            unrealized_pnl = 0.0
            print('PositionI.remove() Warning - position side unrecognized = {}'.format(self.side))

        return unrealized_pnl

    def flatten_inventory(self, order):
        # print(' Flattening {} inventory: {} positions'.format(self.side, self.position_count))
        before_pnl = self.realized_pnl
        [self.remove(order=order) for _ in range(self.position_count)]
        after_pnl = self.realized_pnl
        return after_pnl - before_pnl


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
            is_added = self.long_inventory.add(order=order)
        elif order.side == 'short':
            is_added = self.short_inventory.add(order=order)
        else:
            is_added = False
            print('Broker.add() unknown order.side = %s' % order)
        return is_added

    def remove(self, order):
        if order.side == 'long':
            is_removed = self.long_inventory.remove(order=order)
        elif order.side == 'short':
            is_removed = self.short_inventory.remove(order=order)
        else:
            is_removed = False
            print('Broker.remove() unknown order.side = %s' % order['side'])
        return is_removed

    def get_unrealized_pnl(self, midpoint=100.):
        long_pnl = self.long_inventory.get_unrealized_pnl(midpoint=midpoint)
        short_pnl = self.short_inventory.get_unrealized_pnl(midpoint=midpoint)
        return long_pnl + short_pnl

    def get_realized_pnl(self):
        return self.short_inventory.realized_pnl + self.long_inventory.realized_pnl

    def get_total_pnl(self, midpoint):
        total_pnl = self.get_unrealized_pnl(midpoint=midpoint)
        total_pnl += self.get_realized_pnl()
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

    def step(self, midpoint=100.0):
        self.long_inventory.step(midpoint=midpoint)
        self.short_inventory.step(midpoint=midpoint)

    def get_reward(self, side='long'):
        if side == 'long':
            realized_pnl = self.long_inventory.last_trade['realized_pnl']
            steps_in_position = self.long_inventory.last_trade['steps_in_position']
            drawdown_max = self.long_inventory.last_trade['drawdown_max']
            upside_max = self.long_inventory.last_trade['upside_max']
        elif side == 'short':
            realized_pnl = self.short_inventory.last_trade['realized_pnl']
            steps_in_position = self.short_inventory.last_trade['steps_in_position']
            drawdown_max = self.short_inventory.last_trade['drawdown_max']
            upside_max = self.short_inventory.last_trade['upside_max']
        else:
            realized_pnl = 0.0
            steps_in_position = 0
            drawdown_max = 0.0
            upside_max = 0.0
            print('*gym_trading._get_reward: Unknown order side: {}'.format(side))

        if realized_pnl > 0.0:
            print(' realized_pnl: %.4f | steps_in_position: %i | upside_max: %.4f | drawdown_max: %.4f'
                  % (realized_pnl, steps_in_position, upside_max, drawdown_max))

        return realized_pnl

