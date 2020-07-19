from datetime import datetime as dt
from time import time

import numpy as np
import requests

from configurations import COINBASE_BOOK_ENDPOINT, LOGGER, TIMEZONE
from data_recorder.connector_components.orderbook import OrderBook


class CoinbaseOrderBook(OrderBook):

    def __init__(self, **kwargs):
        """
        Coinbase Order Book constructor.

        :param sym: Instrument or cryptocurrency pair name
        :param db_queue: (optional) queue to connect to the database process
        """
        super(CoinbaseOrderBook, self).__init__(exchange='coinbase', **kwargs)
        self.sequence = 0
        self.diff = 0

    def _get_book(self) -> dict:
        """
        Get order book snapshot.

        :return: order book
        """
        LOGGER.info('%s get_book request made.' % self.sym)
        start_time = time()

        self.clear_book()
        path = (COINBASE_BOOK_ENDPOINT % self.sym)
        book = requests.get(path, params={'level': 3}).json()

        elapsed = time() - start_time
        LOGGER.info('%s get_book request completed in %f seconds.' % (self.sym, elapsed))
        return book

    def load_book(self) -> None:
        """
        Load initial limit order book snapshot.
        """
        book = self._get_book()

        start_time = time()

        self.sequence = book['sequence']
        now = dt.now(tz=TIMEZONE)
        load_time = str(now)

        self.db.new_tick({
            'type': 'load_book',
            'product_id': self.sym,
            'sequence': self.sequence
        })

        for bid in book['bids']:
            msg = {
                'price': float(bid[0]),
                'size': float(bid[1]),
                'order_id': bid[2],
                'side': 'buy',
                'product_id': self.sym,
                'type': 'preload',
                'sequence': self.sequence,
                'time': load_time,
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
                'time': load_time,
            }
            self.db.new_tick(msg)
            self.asks.insert_order(msg)

        self.db.new_tick({
                             'type': 'book_loaded',
                             'product_id': self.sym,
                             'sequence': self.sequence
                         })
        del book
        self.bids.warming_up = self.asks.warming_up = False

        elapsed = time() - start_time
        LOGGER.info('%s: book loaded................in %f seconds' % (self.sym, elapsed))

    def new_tick(self, msg: dict) -> bool:
        """
        Method to process incoming ticks.

        :param msg: incoming tick
        :return: False if there is an exception
        """
        message_type = msg['type']
        if 'sequence' not in msg:
            if message_type == 'subscriptions':
                # request an order book snapshot after the
                #   websocket feed is established
                LOGGER.info('Coinbase Subscriptions successful for : %s' % self.sym)
                self.load_book()
            return True
        elif np.isnan(msg['sequence']):
            # this situation appears during data replays
            #   (and not in live data feeds)
            LOGGER.warn('\n%s found a nan in the sequence' % self.sym)
            return True

        # check the incoming message sequence to verify if there
        # is a dropped/missed message.
        # If so, request a new orderbook snapshot from Coinbase Pro.
        new_sequence = int(msg['sequence'])
        self.diff = new_sequence - self.sequence

        if self.diff == 1:
            # tick sequences increase by an increment of one
            self.sequence = new_sequence
        elif message_type in ['load_book', 'book_loaded', 'preload']:
            # message types used for data replays
            self.sequence = new_sequence
        elif self.diff <= 0:
            if message_type in ['received', 'open', 'done', 'match', 'change']:
                LOGGER.info('%s [%s] has a stale tick: current %i | incoming %i' % (
                    self.sym, message_type, self.sequence, new_sequence))
                return True
            else:
                LOGGER.warn('UNKNOWN-%s %s has a stale tick: current %i | incoming %i' % (
                    self.sym, message_type, self.sequence, new_sequence))
                return True
        else:  # when the tick sequence difference is greater than 1
            LOGGER.info('sequence gap: %s missing %i messages. new_sequence: %i [%s]\n' %
                        (self.sym, self.diff, new_sequence, message_type))
            self.sequence = new_sequence
            return False

        # persist data to Arctic Tick Store
        self.db.new_tick(msg)
        self.last_tick_time = msg.get('time', None)
        # make sure CONFIGS.RECORDING is false when replaying data

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
                self.sell_tracker.add(notional=trade_notional)
                self.bids.match(msg)
                return True
            else:  # trades matched on the asks book are considered buys
                self.buy_tracker.add(notional=trade_notional)
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
            self.bids.warming_up = self.asks.warming_up = False
            LOGGER.info("Book finished loading at {}".format(self.last_tick_time))
            return True

        else:
            LOGGER.warn('\n\n\nunhandled message type\n%s\n\n' % str(msg))
            return False
