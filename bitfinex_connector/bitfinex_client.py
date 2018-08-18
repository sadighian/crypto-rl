from common_components.client import Client
from websockets import ConnectionClosed
# import asyncio
# import threading
# import os


class BitfinexClient(Client):

    def __init__(self, ccy):
        super(BitfinexClient, self).__init__(ccy, 'bitfinex')
        # print('\nBitfinexClient __init__ - Process ID: %s | Thread: %s' % (str(os.getpid()), threading.current_thread().name))

    def run(self):
        """
        Handle incoming level 3 data on a separate thread
        :return:
        """
        # print('\nBitfinexClient run - Process ID: %s | Thread: %s' % (str(os.getpid()), threading.current_thread().name))
        while True:
            msg = self.queue.get()

            if self.book.new_tick(msg) is False:
                print('\n%s missing a tick...going to try and reload the order book\n' % self.sym)
                self.retry_counter += 1

                self.book.bids.done_warming_up = True
                self.book.asks.done_warming_up = True
                print('%s: manually forcing connectionClosed' % self.sym)
                raise ConnectionClosed


# -------------------------------------------------------------------------------------------------------

# """
# This __main__ function is used for testing the
# BitfinexClient class in isolation.
# """
# if __name__ == "__main__":
#     # print('BitfinexClient __main__ - Process ID: %s' % str(os.getpid()))
#     symbols = ['tETHEUR']  #, 'tBCHUSD', 'tETHUSD', 'tLTCUSD']
#     print('Initializing...%s' % symbols)
#     loop = asyncio.get_event_loop()
#     p = dict()
#
#     for sym in symbols:
#         p[sym] = BitfinexClient(sym)
#         p[sym].start()
#         print('Started thread for %s' % sym)
#
#     tasks = asyncio.gather(*[(p[sym].subscribe()) for sym in symbols])
#     print('Gathered %i tasks' % len(symbols))
#
#     try:
#         loop.run_until_complete(tasks)
#         for sym in symbols:
#             p[sym].join()
#             print('Closing [%s]' % p[sym].name)
#         print('loop closed.')
#
#     except KeyboardInterrupt as e:
#         print("Caught keyboard interrupt. Canceling tasks...")
#         tasks.cancel()
#         # loop.run_forever()
#         for sym in symbols:
#             p[sym].join()
#             print('Closing [%s]' % p[sym].name)
#
#     finally:
#         loop.close()
#         print('\nFinally done.')
