from connector_components.client import Client
import websockets
import asyncio


class BitfinexClient(Client):

    def __init__(self, ccy):
        super(BitfinexClient, self).__init__(ccy, 'bitfinex')

    def run(self):
        """
        Handle incoming level 3 data on a separate thread
        :return:
        """
        while True:
            msg = self.queue.get()

            if self.book.new_tick(msg) is False:
                self.retry_counter += 1
                self.book.bids.warming_up = True
                self.book.asks.warming_up = True
                print('\n[Bitfinex - %s] ...going to try and reload the order book\n' % self.sym)
                raise websockets.ConnectionClosed(1006, 'no reason')  # raise an exception to invoke reconnecting


# -------------------------------------------------------------------------------------------------------

# """
# This __main__ function is used for testing the
# BitfinexClient class in isolation.
# """
# if __name__ == "__main__":
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
