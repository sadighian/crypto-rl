import json
import os
from common_components.client import Client
import asyncio


class GdaxClient(Client):

    def __init__(self, ccy):
        super(GdaxClient, self).__init__(ccy, 'gdax')
        self.ws_endpoint = 'wss://ws-feed.gdax.com'
        self.request = json.dumps(dict(type='subscribe', product_ids=[self.sym], channels=['full']))

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
        (or process, depending on implementation)
        :return: void
        """
        print('GdaxClient run - Process ID: %s\nThread: %s' % (str(os.getpid()), self.name))
        while True:
            msg = self.queue.get()

            if self.book.new_tick(msg) is False:
                print('\n%s missing a tick...going to try and reload the order book\n' % self.sym)
                self.book.load_book()
                self.retry_counter += 1
                self.queue.task_done()
                continue

            # print(self.book)
            self.queue.task_done()

# -------------------------------------------------------------------------------------------------------

# """
# This __main__ function is used for testing the
# GdaxClient class in isolation.
# """
# if __name__ == "__main__":
#
#     loop = asyncio.get_event_loop()
#     # symbols = ['BCH-USD', 'ETH-USD', 'LTC-USD', 'BTC-USD']
#     symbols = ['BCH-USD']
#     p = dict()
#
#     print('Initializing...%s' % symbols)
#     for sym in symbols:
#         p[sym] = GdaxClient(sym)
#         p[sym].start()
#
#     tasks = asyncio.gather(*[(p[sym].subscribe()) for sym in symbols])
#     print('Gathered %i tasks' % len(symbols))
#
#     try:
#         loop.run_until_complete(tasks)
#         loop.close()
#         print('loop closed.')
#
#     except KeyboardInterrupt as e:
#         print("Caught keyboard interrupt. Canceling tasks...")
#         tasks.cancel()
#         tasks.exception()
#         for sym in symbols:
#             p[sym].join()
#             print('Closing [%s]' % p[sym].name)
#
#     finally:
#         loop.close()
#         print('\nFinally done.')
