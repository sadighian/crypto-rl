import asyncio
import numpy as np
from threading import Timer
from bitfinex_connector.bitfinex_client import BitfinexClient
from gdax_connector.gdax_client import GdaxClient
from datetime import datetime as dt
from pymongo import MongoClient
from multiprocessing import Process
import time
# import threading
# import os


# Configurations
MONGO_ENDPOINT = 'mongodb://localhost:27017/'
RECORD_DATA = False


class Crypto(Process):

    def __init__(self, symbols):
        super(Crypto, self).__init__()
        self.symbols = symbols
        self.recording = RECORD_DATA  # set to false if you do not want to record market data
        self.db = None
        self.timer_frequency = 0.195  # 0.2 = 5x second
        self.workers = dict()
        self.current_time = dt.now()

    def _add_to_mongo(self, current_time, gdaxClient, bitfinexClient):
        """
        Insert snapshot of limit order book into Mongo DB
        :param current_time: dt.now()
        :return: void
        """
        if self.db[gdaxClient.sym] is not None:
            current_date = current_time.strftime("%Y-%m-%d")
            self.db[gdaxClient.sym][current_date].insert_one({
                'gdax': gdaxClient.render_book(),
                'bitfinex': bitfinexClient.render_book(),
                'time': dt.now()
            })
        else:
            print('%s ---> %s  -  %s' % (gdaxClient.sym, gdaxClient.book, bitfinexClient.book))

    # noinspection PyTypeChecker
    def timer_worker(self, gdaxClient, bitfinexClient):
        """
        Thread worker to be invoked every N seconds
        :return: void
        """
        Timer(self.timer_frequency, self.timer_worker, args=(gdaxClient, bitfinexClient,)).start()
        self.current_time = dt.now()
        if gdaxClient.book.bids.warming_up is False & bitfinexClient.book.bids.warming_up is False:
            self._add_to_mongo(self.current_time, gdaxClient, bitfinexClient)
        else:
            if gdaxClient.book.bids.warming_up:
                print('GDAX - %s is warming up' % gdaxClient.sym)
            if bitfinexClient.book.bids.warming_up:
                print('Bitfinex - %s is warming up' % bitfinexClient.sym)

    # noinspection PyTypeChecker
    def run(self):
        # print('\nCrypto run - Process ID: %s | Thread: %s' % (str(os.getpid()), threading.current_thread().name))
        if self.recording:
            self.db = dict([(sym, MongoClient(MONGO_ENDPOINT)[sym])
                            for sym in list(np.hstack(self.symbols))])
        else:
            self.db = dict([(sym, None) for sym in list(np.hstack(self.symbols))])

        for gdax, bitfinex in zip(*self.symbols):
            self.workers[gdax], self.workers[bitfinex] = GdaxClient(gdax), BitfinexClient(bitfinex)
            self.workers[gdax].start(), self.workers[bitfinex].start()
            # print('Crypto: [%s] & [%s] workers instantiated on process_id %s' % (gdax, bitfinex, str(os.getpid())))
            Timer(5.0, self.timer_worker, args=(self.workers[gdax], self.workers[bitfinex],)).start()

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
            [self.workers[sym].join() for sym in self.workers.keys()]

        finally:
            loop.close()
            print('\nCrypto: Finally done.')


if __name__ == "__main__":
    # print('\n__name__ = __main__ - Process ID: %s | Thread: %s' % (str(os.getpid()), threading.current_thread().name))

    # basket = [['BTC-USD', 'BCH-USD', 'ETH-USD', 'LTC-USD', 'BTC-EUR', 'BCH-EUR', 'ETH-EUR', 'LTC-EUR', 'BTC-GBP'],  # GDAX pairs
    #           ['tBTCUSD', 'tBCHUSD', 'tETHUSD', 'tLTCUSD', 'tBTCEUR', 'tBCHEUR', 'tETHEUR', 'tLTCEUR', 'tBTCGBP']]  # Bitfinex pairs

    basket = [['BTC-GBP', 'BTC-EUR', 'ETH-EUR'],
              ['tBTCGBP', 'tBTCEUR', 'tETHEUR']]

    for gdax, bitfinex in zip(*basket):
        Crypto([[gdax], [bitfinex]]).start()
        time.sleep(2)
        print('\nProcess started up for %s' % gdax)
