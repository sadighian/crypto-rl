from common_components.client import Client
# import asyncio
# import threading
# import os


class GdaxClient(Client):

    def __init__(self, ccy):
        super(GdaxClient, self).__init__(ccy, 'gdax')

    def run(self):
        """
        Handle incoming level 3 data on a separate thread
        :return: void
        """
        while True:
            msg = self.queue.get()

            if self.book.new_tick(msg) is False:
                print('\n%s missing a tick...going to try and reload the order book\n' % self.sym)
                self.book.load_book()
                self.retry_counter += 1
                continue


# -------------------------------------------------------------------------------------------------------

# """
# This __main__ function is used for testing the
# GdaxClient class in isolation.
# """
# if __name__ == "__main__":
#
#     loop = asyncio.get_event_loop()
#     # symbols = ['BCH-USD', 'LTC-USD', 'BTC-USD']
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
#         for sym in symbols:
#             p[sym].join()
#             print('Closing [%s]' % p[sym].name)
#
#     finally:
#         loop.close()
#         print('\nFinally done.')
