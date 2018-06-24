import asyncio
import json
import os
import time
from datetime import datetime as dt
from multiprocessing import JoinableQueue as Queue
from threading import Thread
import websockets
from bitfinex_connector.orderbook import Book


class BitfinexClient(Thread):

    def __init__(self, ccy):
        super(BitfinexClient, self).__init__()
        self.queue = Queue(maxsize=1000)
        self.ws = None
        self.sym = ccy
        self.book = Book(ccy)
        self.retry_counter = 0
        self.last_subscribe_time = None
        self.return_queue = Queue(maxsize=1)

    async def subscribe(self):
        """
        Subscribe to full order book
        :return: void
        """
        try:
            self.ws = await websockets.connect('wss://api.bitfinex.com/ws/2')

            request = json.dumps({
                "event": "subscribe",
                "channel": "book",
                "prec": "R0",
                "freq": "F0",
                "symbol": self.sym,
                "len": "100"
            })
            print('BitfinexClient: %s BOOK subscription request sent:\n%s\n' % (self.sym, request))
            await self.ws.send(request)

            trades_request = json.dumps({
                "event": "subscribe",
                "channel": "trades",
                "symbol": self.sym
            })
            print('BitfinexClient: %s TRADES subscription request sent:\n%s\n' % (self.sym, trades_request))
            await self.ws.send(trades_request)
            self.last_subscribe_time = dt.now()

            while True:
                self.queue.put(json.loads(await self.ws.recv()))

        except websockets.ConnectionClosed as exception:
            print('BitfinexClient: subscription exception %s' % exception)
            self.retry_counter += 1
            elapsed = (dt.now() - self.last_subscribe_time).seconds

            if elapsed < 5:
                sleep_time = max(5 - elapsed, 1)
                time.sleep(sleep_time)
                print('BitfinexClient - %s is sleeping %i seconds...' % (self.sym, sleep_time))

            if self.retry_counter < 30:
                print('BitfinexClient: Retrying to connect... attempted #%i' % self.retry_counter)
                await self.subscribe()  # recursion
            else:
                print('BitfinexClient: %s Ran out of reconnection attempts. Have already tried %i times.'
                      % (self.sym, self.retry_counter))

    async def unsubscribe(self):
        for channel in self.book.channel_id:
            request = {
               "event": "unsubscribe",
               "chanId": channel
            }
            print('BitfinexClient: %s unsubscription request sent:\n%s\n' % (self.sym, request))
            await self.ws.send(request)

    def run(self):
        """
        Handle incoming level 3 data on a separate process
        (or thread, depending on implementation)
        :return:
        """
        print('BitfinexClient Run - Process ID: %s' % str(os.getpid()))
        while True:
            msg = self.queue.get()

            if self.book.new_tick(msg) is False:
                self.subscribe()
                self.retry_counter += 1
                self.queue.task_done()
                continue

            self.queue.task_done()

    def render_book(self):
        return self.book.render_book()

# -------------------------------------------------------------------------------------------------------

"""
This __main__ function is used for testing the
BitfinexClient class in isolation.
"""
if __name__ == "__main__":
    symbols = ['tETHUSD']  #, 'tBCHUSD', 'tETHUSD', 'tLTCUSD']
    print('Initializing...%s' % symbols)
    loop = asyncio.get_event_loop()
    p = dict()

    for sym in symbols:
        p[sym] = BitfinexClient(sym)
        p[sym].start()
        print('[%s] started for [%s]' % (p[sym].name, sym))

    tasks = asyncio.gather(*[(p[sym].subscribe()) for sym in symbols])
    print('Gathered %i tasks' % len(symbols))

    try:
        loop.run_until_complete(tasks)
        loop.close()
        print('loop closed.')

    except KeyboardInterrupt as e:
        print("Caught keyboard interrupt. Canceling tasks...")
        tasks.cancel()
        # loop.run_forever()
        tasks.exception()
        for sym in symbols:
            p[sym].join()
            print('Closing [%s]' % p[sym].name)

    finally:
        loop.close()
        print('\nFinally done.')
