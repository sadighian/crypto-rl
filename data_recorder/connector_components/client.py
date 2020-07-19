import json
import time
from abc import ABC, abstractmethod
from datetime import datetime as dt
from multiprocessing import Queue
from threading import Thread  # , Timer

import websockets

from configurations import LOGGER, MAX_RECONNECTION_ATTEMPTS, TIMEZONE  # , SNAPSHOT_RATE


class Client(Thread, ABC):

    def __init__(self, sym: str, exchange: str):
        """
        Client constructor.

        :param sym: currency symbol
        :param exchange: 'bitfinex' or 'coinbase' or 'bitmex'
        """
        super(Client, self).__init__(name=sym, daemon=True)
        self.sym = sym
        self.exchange = exchange
        self.retry_counter = 0
        self.max_retries = MAX_RECONNECTION_ATTEMPTS
        self.last_subscribe_time = None
        self.last_worker_time = None
        self.queue = Queue(maxsize=0)
        # Attributes that get overridden in sub-classes
        self.ws = None
        self.ws_endpoint = None
        self.request = self.trades_request = None
        self.request_unsubscribe = None
        self.book = None
        LOGGER.info('%s client instantiated.' % self.exchange.upper())

    async def subscribe(self) -> None:
        """
        Subscribe to full order book.
        """
        try:
            self.ws = await websockets.connect(self.ws_endpoint)

            if self.request is not None:
                LOGGER.info('Requesting Book: {}'.format(self.request))
                await self.ws.send(self.request)
                LOGGER.info('BOOK %s: %s subscription request sent.' %
                            (self.exchange.upper(), self.sym))

            if self.trades_request is not None:
                LOGGER.info('Requesting Trades: {}'.format(self.trades_request))
                await self.ws.send(self.trades_request)
                LOGGER.info('TRADES %s: %s subscription request sent.' %
                            (self.exchange.upper(), self.sym))

            self.last_subscribe_time = dt.now(tz=TIMEZONE)

            # Add incoming messages to a queue, which is consumed and processed
            #  in the run() method.
            while True:
                self.queue.put(json.loads(await self.ws.recv()))

        except websockets.ConnectionClosed as exception:
            LOGGER.warn('%s: subscription exception %s' % (self.exchange, exception))
            self.retry_counter += 1
            elapsed = (dt.now(tz=TIMEZONE) - self.last_subscribe_time).seconds

            if elapsed < 10:
                sleep_time = max(10 - elapsed, 1)
                time.sleep(sleep_time)
                LOGGER.info('%s - %s is sleeping %i seconds...' %
                            (self.exchange, self.sym, sleep_time))

            if self.retry_counter < self.max_retries:
                LOGGER.info('%s: Retrying to connect... attempted #%i' %
                            (self.exchange, self.retry_counter))
                await self.subscribe()  # recursion
            else:
                LOGGER.warn('%s: %s Ran out of reconnection attempts. '
                            'Have already tried %i times.' %
                            (self.exchange, self.sym, self.retry_counter))

    async def unsubscribe(self) -> None:
        """
        Unsubscribe limit order book WebSocket from exchange.
        """
        LOGGER.info('Client - %s sending unsubscribe request for %s.' %
                    (self.exchange.upper(), self.sym))

        await self.ws.send(self.request_unsubscribe)
        output = json.loads(await self.ws.recv())

        LOGGER.info('Client - %s: unsubscribe successful.' % (self.exchange.upper()))
        LOGGER.info('unsubscribe() -> Output:')
        LOGGER.info(output)

    @abstractmethod
    def run(self) -> None:
        """
        Thread to override in Coinbase or Bitfinex or Bitmex implementation class.
        """
        LOGGER.info("run() initiated on : {}".format(self.name))
        self.last_worker_time = dt.now()
        # Used for debugging exchanges individually
        # Timer(4.0, _timer_worker, args=(self.book, self.last_worker_time,)).start()

# from data_recorder.connector_components.orderbook import OrderBook

# Used for debugging exchanges individually
# def _timer_worker(orderbook: OrderBook, last_worker_time: dt) -> None:
#     """
#     Thread worker to be invoked every N seconds
#     (e.g., configs.SNAPSHOT_RATE)
#
#     :param orderbook: OrderBook
#     :return: void
#     """
#     now = dt.now()
#     delta = now - last_worker_time
#     print('\n{} - {} with delta {}\n{}'.format(orderbook.sym, now, delta.microseconds,
#                                                orderbook))
#     last_worker_time = now
#
#     Timer(SNAPSHOT_RATE, _timer_worker, args=(orderbook, last_worker_time,)).start()
#
#     if orderbook.done_warming_up:
#         """
#         This is the place to insert a trading model.
#         You'll have to create your own.
#
#         Example:
#             orderbook_data = tuple(coinbaseClient.book, bitfinexClient.book)
#             model = agent.dqn.Agent()
#             fix_api = SomeFixAPI()
#             action = model(orderbook_data)
#             if action is buy:
#                 buy_order = create_order(pair, price, etc.)
#                 fix_api.send_order(buy_order)
#
#         """
#         _ = orderbook.render_book()
