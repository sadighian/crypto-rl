
class Order(object):

    def __init__(self, ccy='BTC-USD', price=None, size=1000.0, side='long'):
        self.sym = ccy
        self.price = price
        self.size = size
        self.side = side
        print('Order() object instantiated:\nccy: %s\nprice: %f\nsize: %f\nside: %s' %
              (self.ccy, self.price, self.size, self.side, self.type))

    def __str__(self):
        return 'ccy: %s | price: %f | size: %f | side: %s' % \
              (self.sym, self.price, self.size, self.side)
