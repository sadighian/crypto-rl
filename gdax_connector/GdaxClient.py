import asyncio
import json
import os
import time
from datetime import datetime as dt
from multiprocessing import JoinableQueue as Queue
from threading import Thread

import websockets

from gdax_connector.orderbook import Book


class GdaxClient(Thread):

    def __init__(self, ccy):
        super(GdaxClient, self).__init__()
        self.queue = Queue()
        self.ws = None
        self.sym = ccy
        self.book = Book(ccy)
        self.retry_counter = 0
        self.last_subscribe_time = None

    async def subscribe(self):
        """
        Subscribe to full order book
        :return: void
        """
        try:
            request = json.dumps(dict(type='subscribe', product_ids=[self.sym], channels=['full']))
            self.ws = await websockets.connect('wss://ws-feed.gdax.com')
            print('GdaxClient: %s request sent:\n%s' % (self.sym, request))

            await self.ws.send(request)
            self.last_subscribe_time = dt.now()

            while True:
                self.queue.put(json.loads(await self.ws.recv()))

        except websockets.ConnectionClosed as exception:
            print('GdaxClient: subscription exception %s' % exception)
            self.retry_counter += 1
            elapsed = (dt.now() - self.last_subscribe_time).seconds

            if elapsed < 5:
                sleep_time = max(5 - elapsed, 1)
                time.sleep(sleep_time)
                print('GdaxClient - %s is sleeping %i seconds...' % (self.sym, sleep_time))

            if self.retry_counter < 30:
                print('GdaxClient: Retrying to connect... attempted #%i' % self.retry_counter)
                await self.subscribe()  # recursion
            else:
                print('GdaxClient: %s Ran out of reconnection attempts. Have already tried %i times.'
                      % (self.sym, self.retry_counter))

    async def unsubscribe(self):
        print('GdaxClient: attempting to unsubscribe from %s' % self.sym)
        request = json.dumps(dict(type='unsubscribe', product_ids=[self.sym], channels=['full']))
        print('GdaxClient: %s request sent:\n%s' % (self.sym, request))
        await self.ws.send(request)
        output = json.loads(await self.ws.recv())
        print('GdaxClient: Unsubscribe successful %s' % output)

    def run(self):
        """
        Handle incoming level 3 data on a separate process
        (or thread, depending on implementation)
        :return: void
        """
        print('GdaxClient Run - Process ID: %s' % str(os.getpid()))
        self.book.load_book()
        while True:
            msg = self.queue.get()

            if 'sequence' not in msg:
                continue

            if self.book.new_tick(msg) is False:
                self.book.load_book()
                self.retry_counter += 1
                self.queue.task_done()
                continue

            self.queue.task_done()


# -------------------------------------------------------------------------------------------------------


if __name__ == "__main__":

    loop = asyncio.get_event_loop()
    # symbols = ['BCH-USD', 'ETH-USD', 'LTC-USD', 'BTC-USD']
    symbols = ['BCH-USD']
    p = dict()

    print('Initializing...%s' % symbols)
    for sym in symbols:
        p[sym] = GdaxClient(sym)
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