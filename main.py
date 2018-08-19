import asyncio
from bitfinex_connector.bitfinex_client import BitfinexClient
from gdax_connector.gdax_client import GdaxClient
from common_components import configs


class Crypto(object):

    def __init__(self, symbols):
        super(Crypto, self).__init__()
        self.symbols = symbols
        self.workers = dict()

    def start(self):
        """
        Processes market data subscription per crypto pair (e.g., BTC-USD)
        :return: void
        """
        for gdax, bitfinex in self.symbols:
            self.workers[gdax], self.workers[bitfinex] = GdaxClient(gdax), BitfinexClient(bitfinex)
            self.workers[gdax].start(), self.workers[bitfinex].start()

        tasks = asyncio.gather(*[self.workers[sym].subscribe() for sym in self.workers.keys()])
        loop = asyncio.get_event_loop()
        print('Crypto Gathered %i tasks' % len(self.workers.keys()))

        try:
            loop.run_until_complete(tasks)
            loop.close()
            [self.workers[sym].join() for sym in self.workers.keys()]
            print('Crypto: loop closed.')

        except KeyboardInterrupt as e:
            print("Crypto: Caught keyboard interrupt. Canceling tasks... %s" % e)
            tasks.cancel()
            loop.close()
            [self.workers[sym].join() for sym in self.workers.keys()]

        finally:
            loop.close()
            print('\nCrypto: Finally done.')


if __name__ == "__main__":
    """
    Entry point of application
    """
    crypto = Crypto(configs.BASKET)
    crypto.start()
