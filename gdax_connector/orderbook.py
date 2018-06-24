from datetime import datetime as dt
from time import time
from common_components.abook import ABook
try:
    import ujson as json
except ImportError:
    import json
from gdax_connector.diction import Diction
import requests
from threading import Timer


class Book(ABook):

    def __init__(self, sym):
        super(Book, self).__init__(sym)
        self.bids = Diction(sym, 'bids')
        self.asks = Diction(sym, 'asks')
        self.sequence = 0

    def __str__(self):
        return '%s  |  %s' % (self.bids, self.asks)

    def clear_book(self):
        """
        Method to reset the limit order book
        :return: void
        """
        self.bids.clear()
        self.asks.clear()

    def render_book(self):
        """
        Convert the limit order book into a dictionary
        :return: dictionary
        """
        return dict({
            'bids': self.bids.get_bids_to_list(),
            'asks': self.asks.get_asks_to_list(),
            'upticks': self._get_trades_tracker['upticks'],
            'downticks': self._get_trades_tracker['downticks'],
            'time': dt.now()
        })

    def render_price(self, side, reference):
        """
        Estimate market order slippage
        :param side: bids or asks
        :param reference: NBBO
        :return: float distance from NBBO
        """
        if side == 'bids':
            return round(self.bids.do_next_price('bids', reference), 2)
        else:
            return round(self.asks.do_next_price('asks', reference), 2)

    def best_bid(self):
        """
        Get the best bid
        :return: float best bid
        """
        return self.bids.get_bid()

    def best_ask(self):
        """
        Get the best ask
        :return: float best ask
        """
        return self.asks.get_ask()

    def get_book(self):
        """
        Get order book snapshot
        :return: order book
        """
        start_time = time()
        self.clear_book()
        path = ('https://api.gdax.com/products/%s/book' % self.sym)
        book = requests.get(path, params={'level': 3}).json()
        elapsed = time() - start_time
        print('get_book: completed %s request in %f seconds.' % (self.sym, elapsed))
        return book

    def load_book(self):
        """
        Load initial limit order book snapshot
        :return: void
        """
        book = self.get_book()
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
        Timer(self.timer_frequency, self.timer_worker).start()

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
        new_sequence = int(msg['sequence'])
        if self.check_sequence(new_sequence):
            return False

        message_type = msg['type']
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
                self.bids.insert_orders(msg['price'], msg['remaining_size'], msg['order_id'], 'buy')
                return True
            else:
                self.asks.insert_orders(msg['price'], msg['remaining_size'], msg['order_id'], 'sell')
                return True

        else:
            print('\n\n\nunhandled message type\n\n\n')
            return False

    def timer_worker(self):
        """
        Thread worker to be invoked every N seconds
        :return: void
        """
        Timer(self.timer_frequency, self.timer_worker).start()
        current_time = dt.now()
        if self.bids.warming_up is False:
            self.record(current_time)
        # diff = (current_time - self.last_time).microseconds
        # print('\nGDAX: %s timer_worker %i for processID %s' % (self.sym, diff - 200000, os.getpid()))
        # self.last_time = current_time

