from connector_components.orderbook import OrderBook
from datetime import datetime as dt
from time import time
import requests
import numpy as np
from configurations.configs import RECORD_DATA


class CoinbaseOrderBook(OrderBook):

    def __init__(self, sym):
        super(CoinbaseOrderBook, self).__init__(sym, 'coinbase')
        self.sequence = 0
        self.diff = 0

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
        book = self._get_book()

        start_time = time()

        self.sequence = book['sequence']
        load_time = str(dt.now(tz=self.db.tz))

        self.db.new_tick({'type': 'load_book',
                          'product_id': self.sym,
                          'sequence': self.sequence})

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

        self.db.new_tick({'type': 'book_loaded',
                          'product_id': self.sym,
                          'sequence': self.sequence})
        del book
        self.bids.warming_up = False
        self.asks.warming_up = False

        elapsed = time() - start_time
        print('%s: book loaded................in %f seconds' % (self.sym, elapsed))

    def _check_sequence(self, new_sequence, message_type):
        """
        Check for gap in incoming tick sequence
        :param new_sequence: incoming tick
        :return: True = reset order book / False = no sequence gap
        """
        self.diff = new_sequence - self.sequence
        if self.diff == 1:
            self.sequence = new_sequence
            return False
        elif self.diff <= 0:
            return False
        elif message_type == 'preload':  # used for simulations
            self.sequence = new_sequence
            return False
        else:
            print('sequence gap: %s missing %i messages. new_sequence: %i [%s]\n' %
                  (self.sym, self.diff, new_sequence, message_type))
            self.sequence = new_sequence
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
            print('\n%s found a nan in the sequence' % self.sym)
            return True

        new_sequence = int(msg['sequence'])
        if self._check_sequence(new_sequence, message_type):
            if message_type == 'load_book':
                self.clear_book()
            return False

        self.db.new_tick(msg)  # make sure CONFIGS.RECORDING is false when replaying data

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
            trade_notional = float(msg['price']) * float(msg['size'])
            if side == 'buy':  # trades matched on the bids book are considered sells
                self.trade_tracker['sells'] += trade_notional
                self.bids.match(msg)
                return True
            else:  # trades matched on the asks book are considered buys
                self.trade_tracker['buys'] += trade_notional
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

        elif message_type == 'book_loaded':
            self.bids.warming_up = False
            self.asks.warming_up = False
            return True

        else:
            print('\n\n\nunhandled message type\n%s\n\n' % str(msg))
            return False
