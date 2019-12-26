from data_recorder.bitfinex_connector.bitfinex_client import BitfinexClient
from data_recorder.coinbase_connector.coinbase_client import CoinbaseClient
from configurations import SNAPSHOT_RATE, BASKET, LOGGER
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

    def run(self) -> None:
        """
        New process created to instantiate limit order books for
            (1) Coinbase Pro, and
            (2) Bitfinex.

        Connections made to each exchange are made asynchronously thanks to asyncio.

        :return: void
        """
        coinbase, bitfinex = self.symbols

        self.workers[coinbase] = CoinbaseClient(sym=coinbase)
        self.workers[bitfinex] = BitfinexClient(sym=bitfinex)

        self.workers[coinbase].start(), self.workers[bitfinex].start()

        Timer(5.0, self.timer_worker,
              args=(self.workers[coinbase], self.workers[bitfinex],)).start()

        tasks = asyncio.gather(*[self.workers[sym].subscribe()
                                 for sym in self.workers.keys()])
        loop = asyncio.get_event_loop()
        LOGGER.info('Recorder: Gathered %i tasks' % len(self.workers.keys()))

        try:
            loop.run_until_complete(tasks)
            loop.close()
            [self.workers[sym].join() for sym in self.workers.keys()]
            LOGGER.info('Recorder: loop closed for %s and %s.' %
                        (coinbase, bitfinex))

        except KeyboardInterrupt as e:
            LOGGER.info("Recorder: Caught keyboard interrupt. \n%s" % e)
            tasks.cancel()
            loop.close()
            [self.workers[sym].join() for sym in self.workers.keys()]

        finally:
            loop.close()
            LOGGER.info('Recorder: Finally done for %s and %s.' %
                        (coinbase, bitfinex))

    def timer_worker(self, coinbaseClient: CoinbaseClient,
                     bitfinexClient: BitfinexClient) -> None:
        """
        Thread worker to be invoked every N seconds (e.g., configurations.SNAPSHOT_RATE)

        :param coinbaseClient: CoinbaseClient
        :param bitfinexClient: BitfinexClient
        :return: void
        """
        Timer(self.timer_frequency, self.timer_worker,
              args=(coinbaseClient, bitfinexClient,)).start()
        self.current_time = dt.now()

        if coinbaseClient.book.done_warming_up & \
                bitfinexClient.book.done_warming_up:
            """
            This is the place to insert a trading model. 
            You'll have to create your own.
            
            Example:
                orderbook_data = tuple(coinbaseClient.book, bitfinexClient.book)
                model = agent.dqn.Agent()
                fix_api = SomeFixAPI() 
                action = model(orderbook_data)
                if action is buy:
                    buy_order = create_order(pair, price, etc.)
                    fix_api.send_order(buy_order)
            
            """
            LOGGER.info('%s >> %s' % (coinbaseClient.sym, coinbaseClient.book))
        else:
            if coinbaseClient.book.done_warming_up:
                LOGGER.info('Coinbase - %s is warming up' % coinbaseClient.sym)
            if bitfinexClient.book.done_warming_up:
                LOGGER.info('Bitfinex - %s is warming up' % bitfinexClient.sym)


def main():
    LOGGER.info('Starting recorder with basket = {}'.format(BASKET))
    for coinbase, bitfinex in BASKET:
        Recorder((coinbase, bitfinex)).start()
        LOGGER.info('Process started up for %s' % coinbase)
        time.sleep(9)


if __name__ == "__main__":
    """
    Entry point of application
    """
    main()
