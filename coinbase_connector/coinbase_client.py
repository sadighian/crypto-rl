from connector_components.client import Client
import asyncio


class CoinbaseClient(Client):

    def __init__(self, ccy):
        super(CoinbaseClient, self).__init__(ccy, 'coinbase')

    def run(self):
        """
        Handle incoming level 3 data on a separate thread
        :return: void
        """
        while True:
            msg = self.queue.get()

            if self.book.new_tick(msg) is False:
                self.book.load_book()
                self.retry_counter += 1
                print('\n[Coinbase - %s] ...going to try and reload the order book\n' % self.sym)
                continue


# -------------------------------------------------------------------------------------------------------

# """
# This __main__ function is used for testing the
# CoinbaseClient class in isolation.
# """
# if __name__ == "__main__":
#
#     loop = asyncio.get_event_loop()
#     symbols = ['BTC-USD']#, 'BCH-USD', 'LTC-USD', 'ETH-USD']
#     p = dict()
#
#     print('Initializing...%s' % symbols)
#     for sym in symbols:
#         p[sym] = CoinbaseClient(sym)
#         p[sym].start()
#
#     tasks = asyncio.gather(*[(p[sym].subscribe()) for sym in symbols])
#     print('Gathered %i tasks' % len(symbols))
#
#     try:
#         loop.run_until_complete(tasks)
#         print('TASK are complete for {}'.format(symbols))
#         loop.close()
#         for sym in symbols:
#             p[sym].join()
#             print('Closing [%s]' % p[sym].name)
#         print('loop closed.')
#
#     except KeyboardInterrupt as e:
#         print("Caught keyboard interrupt. Canceling tasks...")
#         tasks.cancel()
#         loop.close()
#         for sym in symbols:
#             p[sym].join()
#             print('Closing [%s]' % p[sym].name)
#
#     finally:
#         loop.close()
#         print('\nFinally done.')
