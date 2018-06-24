from datetime import datetime as dt
from time import time
import requests
from common_components.abook import ABook


class Book(ABook):

    def __init__(self, sym, exchange):
        super(Book, self).__init__(sym, exchange)
        self.sequence = 0

    def _get_book(self):
        """
        Get order book snapshot
        :return: order book
        """
        start_time = time()
        self.clear_book()
        path = ('https://api.gdax.com/products/%s/book' % self.sym)
        book = requests.get(path, params={'level': 3}).json()
        elapsed = time() - start_time
        print('_get_book: completed %s request in %f seconds.' % (self.sym, elapsed))
        return book

    def load_book(self):
        """
        Load initial limit order book snapshot
        :return: void
        """
        book = self._get_book()
        start_time = time()
        for bid in book['bids']:
            self.bids.insert_order({
                'price': float(bid[0]),
                'size': float(bid[1]),
                'order_id': bid[2],
                'side': 'buy',
                'product_id': self.sym,
                'type': 'preload',
                'time': dt.now()
            })

        for ask in book['asks']:
            self.asks.insert_order({
                'price': float(ask[0]),
                'size': float(ask[1]),
                'order_id': ask[2],
                'side': 'sell',
                'product_id': self.sym,
                'type': 'preload',
                'time': dt.now()
            })

        self.sequence = book['sequence']
        del book

        self.bids.warming_up = False
        self.asks.warming_up = False

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
            size = float(msg['size']) * float(msg['price'])
            if side == 'buy':
                self.bids.match(msg)
                self.trades['downticks']['size'] += size
                self.trades['downticks']['count'] += 1
                print('match: %s --' % str(msg['price']))
                return True
            else:
                self.asks.match(msg)
                self.trades['upticks']['size'] += size
                self.trades['upticks']['count'] += 1
                print('match: %s ++' % str(msg['price']))
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
                self.bids.insert_orders(msg['price'], msg['remaining_size'], msg['order_id'], self.sym, 'buy')
                return True
            else:
                self.asks.insert_orders(msg['price'], msg['remaining_size'], msg['order_id'], self.sym, 'sell')
                return True

        else:
            print('\n\n\nunhandled message type\n\n\n')
            return False
