import json
import time
from datetime import datetime as dt
from multiprocessing import JoinableQueue as Queue
from threading import Thread

import websockets

from bitfinex_connector.orderbook import Book as BitfinexBook
from gdax_connector.orderbook import Book as GdaxBook


class Client(Thread):

    def __init__(self, ccy, exchange):
        super(Client, self).__init__()
        self.book = GdaxBook(ccy, 'gdax') if exchange == 'gdax' else BitfinexBook(ccy)
        self.ws = None
        self.ws_endpoint = ''
        self.sym = ccy
        self.retry_counter = 0
        self.max_retries = 30
        self.last_subscribe_time = None
        self.exchange = exchange
        self.request = None
        self.trades_request = None
        self.queue = Queue(maxsize=1000)

    async def unsubscribe(self):
        pass

    def run(self):
        """
        Handle incoming level 3 data on a separate process
        (or process, depending on implementation)
        :return:
        """
        pass

    async def subscribe(self):
        """
        Subscribe to full order book
        :return: void
        """
        try:
            self.ws = await websockets.connect(self.ws_endpoint)

            if self.request is None:
                print('%s: Request to connect to book websocket is null.' % self.exchange)
            else:
                await self.ws.send(self.request)
                print('BOOK %s: %s subscription request sent.' % (self.exchange, self.sym))
                print(self.request)

            if self.trades_request is None:
                print('%s: Request to connect to trades websocket is null.' % self.exchange)
            else:
                await self.ws.send(self.trades_request)
                print('TRADES %s: %s subscription request sent.' % (self.exchange, self.sym))

            self.last_subscribe_time = dt.now()
            print('set the last subscribed time to %s' % str(self.last_subscribe_time))

            while True:
                self.queue.put(json.loads(await self.ws.recv()))
                # print(self.book)

        except websockets.ConnectionClosed as exception:
            print('%s: subscription exception %s' % (self.exchange, exception))
            self.retry_counter += 1
            elapsed = (dt.now() - self.last_subscribe_time).seconds

            if elapsed < 5:
                sleep_time = max(5 - elapsed, 1)
                time.sleep(sleep_time)
                print('%s - %s is sleeping %i seconds...' % (self.exchange, self.sym, sleep_time))

            if self.retry_counter < self.max_retries:
                print('%s: Retrying to connect... attempted #%i' % (self.exchange, self.retry_counter))
                await self.subscribe()  # recursion
            else:
                print('%s: %s Ran out of reconnection attempts. Have already tried %i times.'
                      % (self.exchange, self.sym, self.retry_counter))

    def render_book(self):
        return self.book.render_book()

