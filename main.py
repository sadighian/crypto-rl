import asyncio
import os
import numpy as np
from threading import Timer
from bitfinex_connector.bitfinex_client import BitfinexClient
from gdax_connector.gdax_client import GdaxClient
from datetime import datetime as dt
from pymongo import MongoClient


class Crypto(object):

    def __init__(self):
        self.symbols = [['BTC-USD',  # Gdax symbols
                         'BCH-USD',
                         'ETH-USD',
                         'LTC-USD'],
                        ['tBTCUSD',  # Bitfinex symbols
                         'tBCHUSD',
                         'tETHUSD',
                         'tLTCUSD']]
        self.recording = False
        if self.recording:
            self.db = dict([(sym, MongoClient('mongodb://localhost:27017')[sym]) for sym in list(np.hstack(self.symbols))])
        else:
            self.db = dict([(sym, None) for sym in list(np.hstack(self.symbols))])
        self.timer_frequency = 0.20  # 0.2 = 5x second
        self.workers = dict()
        self.last_time = dt.now()

    def insert_record(self, current_time, cryptoClient):
        """
        Insert snapshot of limit order book into Mongo DB
        :param current_time: dt.now()
        :return: void
        """
        if self.db is not None:
            current_date = current_time.strftime("%Y-%m-%d")
            self.db[current_date].insert_one(cryptoClient.render_book())
        else:
            print('%s -----> %s' % (current_time, cryptoClient.book))

    def timer_worker(self, gdaxClient, bitfinexClient):
        """
        Thread worker to be invoked every N seconds
        :return: void
        """
        print('\n')
        Timer(self.timer_frequency, self.timer_worker, args=(gdaxClient, bitfinexClient,)).start()
        current_time = dt.now()
        self.last_time = current_time
        if gdaxClient.book.bids.warming_up is False:
            self.insert_record(current_time, gdaxClient)
        if bitfinexClient.book.bids.warming_up is False:
            self.insert_record(current_time, bitfinexClient)

    def do_main(self):
        print('invoking do_main() on %s\n' % str(os.getpid()))

        for gdax, bitfinex in zip(*self.symbols):
            self.workers[gdax], self.workers[bitfinex] = GdaxClient(gdax), BitfinexClient(bitfinex)
            self.workers[gdax].start(), self.workers[bitfinex].start()
            print('[%s] started for [%s] with process_id %s' % (gdax, bitfinex, str(os.getpid())))
            Timer(5.0, self.timer_worker, args=(self.workers[gdax], self.workers[bitfinex],)).start()

        tasks = asyncio.gather(*[self.workers[sym].subscribe() for sym in self.workers.keys()])
        loop = asyncio.get_event_loop()
        print('gdax_process Gathered %i tasks' % len(self.workers.keys()))

        try:
            loop.run_until_complete(tasks)
            loop.close()
            print('gdax_process loop closed.')

        except KeyboardInterrupt as e:
            print("gdax_process Caught keyboard interrupt. Canceling tasks... %s" % e)
            tasks.cancel()
            tasks.exception()

        finally:
            loop.close()
            print('\ngdax_process Finally done.')


if __name__ == "__main__":
    print('Starting up...__main__ Process ID: %s\n' % str(os.getpid()))
    crypto = Crypto()
    crypto.do_main()

