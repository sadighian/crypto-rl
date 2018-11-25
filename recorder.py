from bitfinex_connector.bitfinex_client import BitfinexClient
from coinbase_connector.coinbase_client import CoinbaseClient
from configurations.configs import SNAPSHOT_RATE, BASKET
from threading import Timer
from datetime import datetime as dt
from multiprocessing import Process
import time
import asyncio


class Recorder(Process):

    def __init__(self, symbols):
        """
        Constructor of Recorder.

        :param symbols: basket of securities to record...
                        Example: symbols = [('BTC-USD, 'tBTCUSD')]
        """
        super(Recorder, self).__init__()
        self.symbols = symbols
        self.timer_frequency = SNAPSHOT_RATE
        self.workers = dict()
        self.current_time = dt.now()
        self.daemon = False

    # noinspection PyTypeChecker
    def run(self):
        """
        New process created to instantiate limit order books for (1) Coinbase Pro, and (2) Bitfinex.
        Connections made to each exchange are made asynchronously thanks to asyncio.

        :return: void
        """
        coinbase, bitfinex = self.symbols

        self.workers[coinbase], self.workers[bitfinex] = CoinbaseClient(coinbase), BitfinexClient(bitfinex)
        self.workers[coinbase].start(), self.workers[bitfinex].start()
        Timer(5.0, self.timer_worker, args=(self.workers[coinbase], self.workers[bitfinex],)).start()

        tasks = asyncio.gather(*[self.workers[sym].subscribe() for sym in self.workers.keys()])
        loop = asyncio.get_event_loop()
        print('Recorder: Gathered %i tasks' % len(self.workers.keys()))

        try:
            loop.run_until_complete(tasks)
            loop.close()
            [self.workers[sym].join() for sym in self.workers.keys()]
            print('Recorder: loop closed for %s and %s.' % (coinbase, bitfinex))

        except KeyboardInterrupt as e:
            print("Recorder: Caught keyboard interrupt. \n%s" % e)
            tasks.cancel()
            loop.close()
            [self.workers[sym].join() for sym in self.workers.keys()]

        finally:
            loop.close()
            print('\nRecorder: Finally done for %s and %s.' % (coinbase, bitfinex))

    # noinspection PyTypeChecker
    def timer_worker(self, coinbaseClient, bitfinexClient):
        """
        Thread worker to be invoked every N seconds (e.g., configs.SNAPSHOT_RATE)

        :param coinbaseClient: from coinbase_connector.coinbase_client import CoinbaseClient
        :param bitfinexClient: from bitfinex_connector.bitfinex_client import BitfinexClient
        :return: void
        """
        Timer(self.timer_frequency, self.timer_worker, args=(coinbaseClient, bitfinexClient,)).start()
        self.current_time = dt.now()

        if coinbaseClient.book.done_warming_up() & bitfinexClient.book.done_warming_up():
            print('%s >> %s' % (coinbaseClient.sym, coinbaseClient.book))
        else:
            if coinbaseClient.book.done_warming_up():
                print('Coinbase - %s is warming up' % coinbaseClient.sym)
            if bitfinexClient.book.done_warming_up():
                print('Bitfinex - %s is warming up' % bitfinexClient.sym)


def main():
    for coinbase, bitfinex in BASKET:
        Recorder((coinbase, bitfinex)).start()
        print('\nProcess started up for %s' % coinbase)
        time.sleep(9)


if __name__ == "__main__":
    """
    Entry point of application
    """
    main()
