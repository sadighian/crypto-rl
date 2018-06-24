import asyncio
import json
import os
from common_components.aclient import AClient


class GdaxClient(AClient):

    def __init__(self, ccy):
        super(GdaxClient, self).__init__(ccy, 'gdax')
        self.ws_endpoint = 'wss://ws-feed.gdax.com'
        self.request = json.dumps(dict(type='subscribe', product_ids=[self.sym], channels=['full']))
        print('GdaxClient instantiated on Process ID: %s' % str(os.getpid()))

    async def unsubscribe(self):
        print('GdaxClient: attempting to unsubscribe from %s' % self.sym)
        request = json.dumps(dict(type='unsubscribe', product_ids=[self.sym], channels=['full']))
        print('GdaxClient: %s request sent:\n%s' % (self.sym, request))
        await self.ws.send(request)
        output = json.loads(await self.ws.recv())
        print('GdaxClient: Unsubscribe successful %s' % output)

    def on_message(self, queue, return_queue):
        """
        Handle incoming level 3 data on a separate process
        (or process, depending on implementation)
        :return: void
        """
        print('GdaxClient on_message - Process ID: %s' % str(os.getpid()))
        while True:
            msg = queue.get()

            if self.book.new_tick(msg) is False:
                print('missing a tick')
                self.retry_counter += 1
                return_queue.put(self.book)
                queue.task_done()
                self.on_message(queue, return_queue)
                continue

            return_queue.put(self.book)
            queue.task_done()

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
