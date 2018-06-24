import asyncio
import json
import os
from common_components.aclient import AClient


class BitfinexClient(AClient):

    def __init__(self, ccy):
        super(BitfinexClient, self).__init__(ccy, 'bitfinex')
        self.ws_endpoint = 'wss://api.bitfinex.com/ws/2'
        self.request = json.dumps({
            "event": "subscribe",
            "channel": "book",
            "prec": "R0",
            "freq": "F0",
            "symbol": self.sym,
            "len": "100"
        })
        self.trades_request = json.dumps({
            "event": "subscribe",
            "channel": "trades",
            "symbol": self.sym
        })
        print('BitfinexClient instantiated on Process ID: %s' % str(os.getpid()))

    async def unsubscribe(self):
        for channel in self.book.channel_id:
            request = {
               "event": "unsubscribe",
               "chanId": channel
            }
            print('BitfinexClient: %s unsubscription request sent:\n%s\n' % (self.sym, request))
            await self.ws.send(request)

    def on_message(self, queue, return_queue):
        """
        Handle incoming level 3 data on a separate process
        (or process, depending on implementation)
        :return:
        """
        print('BitfinexClient on_message - Process ID: %s' % str(os.getpid()))
        while True:
            msg = queue.get()

            if self.book.new_tick(msg) is False:
                self.subscribe()
                self.retry_counter += 1
                return_queue.put(self.book)
                queue.task_done()
                continue

            return_queue.put(self.book)
            queue.task_done()


# -------------------------------------------------------------------------------------------------------

# """
# This __main__ function is used for testing the
# BitfinexClient class in isolation.
# """
# if __name__ == "__main__":
#     print('BitfinexClient __main__ - Process ID: %s' % str(os.getpid()))
#     symbols = ['tETHUSD']  #, 'tBCHUSD', 'tETHUSD', 'tLTCUSD']
#     print('Initializing...%s' % symbols)
#     loop = asyncio.get_event_loop()
#     p = dict()
#
#     for sym in symbols:
#         p[sym] = BitfinexClient(sym)
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
#         # loop.run_forever()
#         tasks.exception()
#         for sym in symbols:
#             p[sym].join()
#             print('Closing [%s]' % p[sym].name)
#
#     finally:
#         loop.close()
#         print('\nFinally done.')
