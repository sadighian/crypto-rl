from abc import ABC, abstractmethod
from multiprocessing import JoinableQueue as Queue
from threading import Thread
from bitfinex_connector.orderbook import Book as BitfinexBook
from gdax_connector.orderbook import Book as GdaxBook

import websockets
from datetime import datetime as dt
import json
import time


class AClient(ABC):

    def __init__(self, ccy, exchange):
        self.book = GdaxBook(ccy) if exchange == 'gdax' else BitfinexBook(ccy)
        self.ws = None
        self.sym = ccy
        self.retry_counter = 0
        self.max_retries = 30
        self.last_subscribe_time = None
        self.queue, self.return_queue = Queue(), Queue()
        self.process = Thread(target=self.on_message, args=(self.queue, self.return_queue,))
        self.process.name = '[%s-%s]' % (exchange, ccy)
        self.process.daemon = False
        self.exchange = exchange
        self.request = None
        self.trades_request = None

    @abstractmethod
    async def unsubscribe(self):
        pass

    @abstractmethod
    def on_message(self, queue, return_queue):
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
                self.book = self.return_queue.get()
                self.return_queue.task_done()

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

    def start(self):
        self.process.start()
        print('started %s for %s' % (self.name, self.sym))

    def join(self):
        self.process.join(timeout=5.0)
        print('%s: joined %s process' % (self.exchange, self.name))

    @property
    def name(self):
        return self.process.name
