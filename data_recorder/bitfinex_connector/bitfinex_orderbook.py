from time import time
import numpy as np
from data_recorder.connector_components.orderbook import OrderBook


class BitfinexOrderBook(OrderBook):

    def __init__(self, sym):
        super(BitfinexOrderBook, self).__init__(sym, 'bitfinex')
        self.channel_id = {
            'book': int(0),
            'trades': int(0)
        }

    def _load_book(self, book):
        """
        Load initial limit order book snapshot
        :param book: order book snapshot
        :return: void
        """
        start_time = time()

        self.db.new_tick({'type': 'load_book',
                          'product_id': self.sym})

        for row in book[1]:
            msg = {
                "order_id": int(row[0]),
                "price": float(row[1]),
                "size": float(abs(row[2])),
                "side": 'sell' if float(row[2]) < float(0) else 'buy',
                "product_id": self.sym,
                "type": 'preload'
            }
            self.db.new_tick(msg)

            if msg['side'] == 'buy':
                self.bids.insert_order(msg)
            else:
                self.asks.insert_order(msg)

        self.db.new_tick({'type': 'book_loaded',
                          'product_id': self.sym})

        self.bids.warming_up = False
        self.asks.warming_up = False

        elapsed = time() - start_time
        print('%s: book loaded..............in %f seconds\n' % (self.sym, elapsed))

    def new_tick(self, msg):
        """
        Method to process incoming ticks.
        :param msg: incoming tick
        :return: False if there is an exception
        """
        # check for data messages, which only come in lists
        if isinstance(msg, list):
            if msg[0] == self.channel_id['book']:
                return self._process_book(msg)
            elif msg[0] == self.channel_id['trades']:
                return self._process_trades(msg)

        # non-data messages
        elif isinstance(msg, dict):
            if 'event' in msg:
                return self._process_events(msg)
            elif msg['type'] == 'te':
                return self._process_trades_replay(msg)
            elif msg['type'] in ['update', 'preload']:
                return self._process_book_replay(msg)
            elif msg['type'] == 'load_book':
                self.clear_book()
                return True
            elif msg['type'] == 'book_loaded':
                self.bids.warming_up = False
                self.asks.warming_up = False
                return True
            else:
                print('new_tick() message does not know how to be processed = %s' % str(msg))

        # unhandled exception
        else:
            print('unhandled exception\n%s\n' % msg)
            return True

    def _process_book(self, msg):
        """
        Internal method to process FULL BOOK market data
        :param msg: incoming tick
        :return: False if resubscription in required
        """
        # check for a heartbeat
        if msg[1] == 'hb':
            # render_book('heart beat %s' % msg)
            return True

        # order book message (initial snapshot)
        elif np.shape(msg[1])[0] > 3:
            print('%s loading book...' % self.sym)
            self.clear_book()
            self._load_book(msg)
            return True

        else:
            # else, the incoming message is a order update
            order = {
                "order_id": int(msg[1][0]),
                "price": float(msg[1][1]),
                "size": float(abs(msg[1][2])),
                "side": 'sell' if float(msg[1][2]) < float(0) else 'buy',
                "product_id": self.sym,
                "type": 'update'
            }

            self.db.new_tick(order)

            # order should be removed from the book
            if order['price'] == float(0):
                if order['side'] == 'buy':
                    self.bids.remove_order(order)
                elif order['side'] == 'sell':
                    self.asks.remove_order(order)

            # order is a new order or size update for bids
            elif order['side'] == 'buy':
                if order['order_id'] in self.bids.order_map:
                    self.bids.change(order)
                else:
                    self.bids.insert_order(order)

            # order is a new order or size update for asks
            elif order['side'] == 'sell':
                if order['order_id'] in self.asks.order_map:
                    self.asks.change(order)
                else:
                    self.asks.insert_order(order)

            # unhandled msg
            else:
                print('\nUnhandled list msg %s' % msg)

            return True

    def _process_book_replay(self, order):
        """
        Internal method to process FULL BOOK market data
        :param msg: incoming tick
        :return: False if resubscription in required
        """
        # clean up the datatypes
        order['price'] = float(order['price'])
        order['size'] = float(order['size'])

        if order['type'] == 'update':
            # order should be removed from the book
            if order['price'] == float(0):
                if order['side'] == 'buy':
                    self.bids.remove_order(order)
                elif order['side'] == 'sell':
                    self.asks.remove_order(order)
            # order is a new order or size update for bids
            elif order['side'] == 'buy':
                if order['order_id'] in self.bids.order_map:
                    self.bids.change(order)
                else:
                    self.bids.insert_order(order)
            # order is a new order or size update for asks
            elif order['side'] == 'sell':
                if order['order_id'] in self.asks.order_map:
                    self.asks.change(order)
                else:
                    self.asks.insert_order(order)
            # unhandled tick message
            else:
                print('_process_book_replay: unhandled message\n%s' % str(order))

        elif order['type'] == 'preload':
            if order['side'] == 'buy':
                self.bids.insert_order(order)
            else:
                self.asks.insert_order(order)

        elif order['type'] == 'te':
            trade_notional = order['price'] * order['size']
            if order['side'] == 'upticks':
                self.trade_tracker['buys'] += trade_notional
            else:
                self.trade_tracker['sells'] += trade_notional

        else:
            print('\n_process_book_replay() Unhandled list msg %s' % order)

        return True

    def _process_trades(self, msg):
        """
        Internal method to process trade messages
        :param msg: incoming tick
        :return: False if resubscription is required
        """
        if len(msg) == 2:
            #  historical trades
            return True

        msg_type = msg[1]

        if msg[2][2] > 0.0:
            side = 'upticks'
        else:
            side = 'downticks'

        if msg_type == 'hb':
            print('Heartbeat for trades')

        elif msg_type == 'te':
            trade = {
                'price': float(msg[2][3]),
                'size': float(msg[2][2]),
                'side': side,
                'type': msg_type,
                "product_id": self.sym
            }
            self.db.new_tick(trade)
            # print('%s %f' % (side, msg[2][3]))

        # elif msg_type == 'tu':
        #     self.trades[side]['size'] += abs(msg[2][3] * msg[2][2])
        #     self.trades[side]['count'] += 1
        #     print('tu message %s' % msg)

        return True

    def _process_trades_replay(self, msg):
        trade_notional = abs(msg['price'] * msg['size'])

        if msg['side'] == 'upticks':
            self.trade_tracker['buys'] += trade_notional
        else:
            self.trade_tracker['sells'] += trade_notional

        return True

    def _process_events(self, msg):
        """
        Internal method for return code processing
        :param msg: incoming message from websocket
        :return: False if subscription is required
        """
        if msg['event'] == 'subscribed':
            self.channel_id[msg['channel']] = msg['chanId']
            print('%s Added channel_id: %i for %s' % (self.sym, msg['chanId'], msg['channel']))
            return True

        elif msg['event'] == 'info':

            if 'code' in msg:
                code = msg['code']
            else:
                code = None

            if code == 20051:
                print('\nBitfinex - %s: 20051 Stop/Restart Websocket Server (please reconnect)' % self.sym)
                return False  # need to re-subscrbe to the data feed
            elif code == 20060:
                print('\nBitfinex - ' + self.sym + ': 20060 Entering in Maintenance mode. ' +
                      'Please pause any activity and resume after receiving the ' +
                      'info message 20061 (it should take 120 seconds at most).')
                return True
            elif code == 20061:
                print('\nBitfinex - ' + self.sym + ': 20061 Maintenance ended. ' +
                      'You can resume normal activity. ' +
                      'It is advised to unsubscribe/subscribe again all channels.')
                return False  # need to re-subscrbe to the data feed
            elif code == 10300:
                print('\nBitfinex - %s: 10300 Subscription failed (generic)' % self.sym)
                return True
            elif code == 10301:
                print('\nBitfinex - %s: 10301 Already subscribed' % self.sym)
                return True
            elif code == 10302:
                print('\nBitfinex - %s: 10302 Unknown channel' % self.sym)
                return True
            elif code == 10400:
                print('\nBitfinex - %s: 10400 Subscription failed (generic)' % self.sym)
                return True
            elif code == 10401:
                print('\nBitfinex - %s: 10401 Not subscribed' % self.sym)
                return True

