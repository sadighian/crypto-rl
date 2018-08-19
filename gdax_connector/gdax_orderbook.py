from datetime import datetime as dt
from time import time
import requests
from common_components.orderbook import OrderBook


class GdaxOrderBook(OrderBook):

    def __init__(self, sym):
        super(GdaxOrderBook, self).__init__(sym, 'gdax')
        self.sequence = 0

    def _get_book(self):
        """
        Get order book snapshot
        :return: order book
        """
        print('%s get_book request made.' % self.sym)
        start_time = time()
        path = ('https://api.pro.coinbase.com/products/%s/book' % self.sym)
        book = requests.get(path, params={'level': 3}).json()
        elapsed = time() - start_time
        print('%s get_book request completed in %f seconds.' % (self.sym, elapsed))
        return book

    def load_book(self):
        """
        Load initial limit order book snapshot
        :return: void
        """
        book = self._get_book()
        start_time = time()
        self.sequence = book['sequence']
        load_time = str(dt.now(tz=self.db.tz))
        self.db.new_tick({'type': 'load_book'})
        for bid in book['bids']:
            msg = {
                'price': float(bid[0]),
                'size': float(bid[1]),
                'order_id': bid[2],
                'side': 'buy',
                'product_id': self.sym,
                'type': 'preload',
                'sequence': self.sequence,
                'time': load_time
            }
            self.db.new_tick(msg)

        for ask in book['asks']:
            msg = {
                'price': float(ask[0]),
                'size': float(ask[1]),
                'order_id': ask[2],
                'side': 'sell',
                'product_id': self.sym,
                'type': 'preload',
                'sequence': self.sequence,
                'time': load_time
            }
            self.db.new_tick(msg)

        del book

        elapsed = time() - start_time
        print('%s: book loaded................in %f seconds' % (self.sym, elapsed))

    def check_sequence(self, new_sequence):
        """
        Check for gap in incoming tick sequence
        :param new_sequence: incoming tick
        :return: True = reset order book / False = no sequence gap
        """
        diff = new_sequence - self.sequence
        if diff == 1:
            self.sequence = new_sequence
            return False
        elif diff <= 0:
            return False
        else:
            print('sequence gap: %s missing %i messages.\n' % (self.sym, diff))
            return True

    def new_tick(self, msg):
        """
        Method to process incoming ticks.
        :param msg: incoming tick
        :return: False if there is an exception
        """
        message_type = msg['type']
        if 'sequence' not in msg:
            if message_type == 'subscriptions':
                print('GDAX Subscriptions successful for : %s' % self.sym)
                self.load_book()
            return True

        new_sequence = int(msg['sequence'])
        if self.check_sequence(new_sequence):
            return False

        self.db.new_tick(msg)
