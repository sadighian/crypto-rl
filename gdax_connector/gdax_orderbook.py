from datetime import datetime as dt
from time import time
import requests
from common_components.orderbook import OrderBook
import pytz as tz


class GdaxOrderBook(OrderBook):

    def __init__(self, sym):
        super(GdaxOrderBook, self).__init__(sym, 'gdax')
        self.sequence = 0

    def render_book(self):
        return dict(list(self.bids.get_bids_to_list().items()) + list(self.asks.get_asks_to_list().items()))

    def _get_book(self):
        """
        Get order book snapshot
        :return: order book
        """
        print('%s get_book request made.' % self.sym)
        start_time = time()
        self.clear_book()
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
        self.bids.warming_up = True
        self.asks.warming_up = True
        book = self._get_book()
        start_time = time()
        self.sequence = book['sequence']
        load_time = str(dt.now(tz=tz.utc))
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
            self.bids.insert_order(msg)

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
            self.asks.insert_order(msg)

        del book

        self.bids.warming_up = False
        self.asks.warming_up = False

        elapsed = time() - start_time
        print('%s: book loaded................in %f seconds' % (self.sym, elapsed))

    def check_sequence(self, diff):
        """
        Check for gap in incoming tick sequence
        :param new_sequence: incoming tick
        :return: True = reset order book / False = no sequence gap
        """
        if diff <= 1:
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
        diff = new_sequence - self.sequence
        if self.check_sequence(diff):
            return False

        if diff < 0:  # filter out stale ticks
            print('%s has an obsolete tick [incoming=%i] [current=%i]' % (self.sym, new_sequence, self.sequence))
            return True

        self.sequence = new_sequence

        side = msg['side']
        if message_type == 'received':
            return True

        elif message_type == 'open':
            if side == 'buy':
                self.bids.insert_order(msg)
                return True
            else:
                self.asks.insert_order(msg)
                return True

        elif message_type == 'done':
            if side == 'buy':
                self.bids.remove_order(msg)
                return True
            else:
                self.asks.remove_order(msg)
                return True

        elif message_type == 'match':
            if side == 'buy':
                self.bids.match(msg)
                return True
            else:
                self.asks.match(msg)
                return True

        elif message_type == 'change':
            if side == 'buy':
                self.bids.change(msg)
                return True
            else:
                self.asks.change(msg)
                return True

        elif message_type == 'preload':
            if side == 'buy':
                self.bids._insert_orders(msg['price'], msg['remaining_size'], msg['order_id'], self.sym, 'buy')
                return True
            else:
                self.asks._insert_orders(msg['price'], msg['remaining_size'], msg['order_id'], self.sym, 'sell')
                return True

        else:
            print('\n\n\nunhandled message type\n\n\n')
            return False
