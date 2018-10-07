from datetime import datetime as dt
from time import time
import requests
from common_components.orderbook import OrderBook
import numpy as np


class CoinbaseOrderBook(OrderBook):

    def __init__(self, sym):
        super(CoinbaseOrderBook, self).__init__(sym, 'coinbase')
        self.sequence = 0

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
        load_time = str(dt.now(tz=self.db.tz))
        self.db.new_tick({'type': 'load_book', 'product_id': self.sym})
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
            self.db.new_tick(msg)
            self.asks.insert_order(msg)

        del book

        self.bids.warming_up = False
        self.asks.warming_up = False

        elapsed = time() - start_time
        print('%s: book loaded................in %f seconds' % (self.sym, elapsed))

    def check_sequence(self, new_sequence, message_type):
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
        elif message_type == 'preload':  # used for simulations
            self.sequence = new_sequence
        else:
            # print('sequence gap: %s missing %i messages.\n' % (self.sym, diff))
            print('\nBad sequence')
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
                print('Coinbase Subscriptions successful for : %s' % self.sym)
                self.load_book()
            return True
        elif np.isnan(msg['sequence']):
            print('\nfound a nan in the sequence')
            return True

        new_sequence = int(msg['sequence'])
        if self.check_sequence(new_sequence, message_type):
            return False

        self.db.new_tick(msg)

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
                self.bids.insert_order(msg)
                return True
            else:
                self.asks.insert_order(msg)
                return True

        elif message_type == 'load_book':
            self.clear_book()
            return True

        else:
            print('\n\n\nunhandled message type\n\n\n')
            return False
