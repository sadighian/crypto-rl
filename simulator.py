from datetime import datetime as dt
from datetime import timedelta
from multiprocessing import cpu_count, Process, Queue
from arctic import Arctic, TICK_STORE
from arctic.date import DateRange
from coinbase_connector.coinbase_orderbook import CoinbaseOrderBook
from bitfinex_connector.bitfinex_orderbook import BitfinexOrderBook
from configurations.configs import TIMEZONE, MONGO_ENDPOINT, ARCTIC_NAME, RECORD_DATA, MAX_BOOK_ROWS
from sortedcontainers import SortedDict
from dateutil.parser import parse
import numpy as np
import pandas as pd


class Simulator(object):

    def __init__(self):
        try:
            print('Attempting to connect to Arctic...')
            self.arctic = Arctic(MONGO_ENDPOINT)
            self.arctic.initialize_library(ARCTIC_NAME, lib_type=TICK_STORE)
            self.library = self.arctic[ARCTIC_NAME]
            self.queue, self.return_queue = Queue(maxsize=0), Queue(maxsize=0)
            self.number_of_workers = cpu_count()
            self.workers = [Process(name='Process-%i' % num,
                                    args=(self.queue, self.return_queue,),
                                    target=self._do_work) for num in range(self.number_of_workers)]
            print('Connected to Arctic.')
        except Exception as ex:
            self.arctic, self.library, self.workers = None, None, None
            print('Unable to connect to Arctic database')
            print(ex)

    def _start(self, query):
        cursor = self._query_arctic(**query)

        if cursor is None:
            print('\nNothing returned from Arctic for the query: %s' % str(query))
            print('...Exiting.')
            return

        self._split_cursor_and_add_to_queue(cursor)

        del cursor  # clear up resources

        for worker in self.workers:
            worker.start()
            print('Started %s' % worker.name)

    def _stop(self):
        for worker in self.workers:
            worker.join()
            print('Stopped %s' % worker.name)

    def _query_arctic(self, ccy, start_date, end_date):
        print('\nGetting %s tick data from Arctic Tick Store...' % ccy)
        if self.library is None:
            print('exiting from Simulator... no database to query')
            return None

        start_time = dt.now(TIMEZONE)

        cursor = self.library.read(symbol=ccy, date_range=DateRange(start_date, end_date))
        min_datetime = cursor.loc[cursor.type == 'load_book'].index[0]
        cursor = cursor.loc[cursor.index >= min_datetime].copy()

        elapsed = (dt.now(TIMEZONE) - start_time).seconds
        print('Completed querying %i %s records in %i seconds' % (cursor.shape[0], ccy, elapsed))
        return cursor

    def _split_cursor_and_add_to_queue(self, cursor):
        start_time = dt.now(TIMEZONE)

        interval_length = int(cursor.shape[0]/self.number_of_workers)
        starting_index = 0
        ending_index = interval_length
        print('\nSplitting %i records into %i chunks... %i records/cpu' %
              (cursor.shape[0], self.number_of_workers, interval_length))

        for cpu in range(self.number_of_workers):
            if cpu == (self.number_of_workers - 1):
                tmp = cursor.iloc[starting_index:].copy()
                self.queue.put((cpu, tmp))
            else:
                tmp = cursor.iloc[starting_index:ending_index].copy()
                self.queue.put((cpu, tmp))

            starting_index = ending_index
            ending_index = starting_index + interval_length

        elapsed = (dt.now(TIMEZONE) - start_time).seconds
        print('Complete splitting pandas into chunks in %i seconds.' % elapsed)

    @staticmethod
    def _do_work(queue, return_queue):
        start_time = dt.now(TIMEZONE)

        cpu, panda = queue.get()
        results = panda.to_dict('records')
        return_queue.put((cpu, results))

        elapsed = (dt.now(TIMEZONE) - start_time).seconds
        print('Completed. Process-%i dataframe.to_dict() in %i seconds' % (cpu, elapsed))

    def get_tick_history(self, query):
        start_time = dt.now(TIMEZONE)

        assert RECORD_DATA is False

        tick_history_dict = SortedDict()
        counter = 0

        self._start(query)

        while True:
            cpu, list_of_dicts = self.return_queue.get()
            tick_history_dict[cpu] = list_of_dicts
            counter += 1

            if counter == self.number_of_workers:
                tick_history_list = list(sum(tick_history_dict.values()[:], []))
                del tick_history_dict  # free up ram
                print('\n*****\nSuccessfully got data from the return queue\n*****\n')
                break

        self._stop()

        elapsed = (dt.now(TIMEZONE) - start_time).seconds
        print('Completed get_tick_history() in %i seconds' % elapsed)
        return tick_history_list

    @staticmethod
    def get_orderbook_snapshot_history(coinbase_order_book, bitfinex_order_book, tick_history):
        start_time = dt.now(TIMEZONE)

        print('Starting get_orderbook_snapshot_history() for %s and %s' %
              (coinbase_order_book.sym, bitfinex_order_book.sym))

        coinbase_counter = 0
        # bitfinex_counter = 0
        snapshot_list = list()
        last_snapshot_time = None

        for tick in tick_history:

            # determine if incoming tick is from coinbase or bitfinex
            coinbase = True if tick['product_id'] == coinbase_order_book.sym else False

            if 'type' not in tick:
                # filter out bad ticks
                continue

            if tick['type'] == 'load_book':
                # flag for a order book reset
                if coinbase:
                    coinbase_order_book.new_tick(tick)
                else:
                    bitfinex_order_book.new_tick(tick)
                # skip to next loop
                continue

            if tick['type'] == 'preload':
                # flag for order book downloads after a reset
                if coinbase:
                    coinbase_order_book.new_tick(tick)
                    if not coinbase_order_book.done_warming_up():
                        # coinbase tick & order book was just reset
                        # note: logic is slightly different here for simulations
                        coinbase_order_book.bids.warming_up = False
                        coinbase_order_book.asks.warming_up = False

                else:  # bitfinex
                    bitfinex_order_book.new_tick(tick)
                    if not bitfinex_order_book.done_warming_up():
                        # bitfinex tick & order book was just reset
                        # note: logic is slightly different here for simulations
                        bitfinex_order_book.bids.warming_up = False
                        bitfinex_order_book.asks.warming_up = False
                # skip to next loop
                continue

            if coinbase:
                if coinbase_order_book.done_warming_up():
                    new_tick_time = parse(tick['time'])  # timestamp for incoming tick
                    coinbase_counter += 1
                    coinbase_order_book.new_tick(tick)
            elif bitfinex_order_book.done_warming_up():
                # bitfinex_counter += 1
                bitfinex_order_book.new_tick(tick)

            if coinbase_counter == 0:
                # filter out ticks that occur before Coinbase data is ready
                # skip to next loop
                continue
            elif coinbase_counter == 1:
                # start tracking snapshot timestamps
                # and keep in mind that snapshots are tethered to coinbase timestamps
                last_snapshot_time = new_tick_time
                print('%s first tick: %s | Sequence: %i' %
                      (coinbase_order_book.sym, str(new_tick_time), coinbase_order_book.sequence))
                # skip to next loop
                continue

            diff = (new_tick_time - last_snapshot_time).microseconds
            multiple = int(diff / 250000)  # 250000 is 250 milliseconds, or 4x a second

            if multiple >= 1:  # if there is a pause in incoming data, continue to create order book snapshots
                if coinbase_order_book.done_warming_up() & bitfinex_order_book.done_warming_up():
                    coinbase_order_book_snapshot = coinbase_order_book.render_book()
                    bitfinex_order_book_snapshot = bitfinex_order_book.render_book()
                    for _ in range(multiple):
                        snapshot_list.append(list(np.hstack((new_tick_time,
                                                             coinbase_order_book_snapshot,
                                                             bitfinex_order_book_snapshot))))
                        last_snapshot_time += timedelta(milliseconds=250)
                else:
                    last_snapshot_time += timedelta(milliseconds=250)

            if coinbase_counter % 50000 == 0:
                elapsed = (dt.now(TIMEZONE) - start_time).seconds
                print('Completed %i loops in %i seconds' % (coinbase_counter, elapsed))
                if coinbase_counter % 250000 == 0:
                    # TODO: Remove this snippet for PROD - only here now for testing purposes
                    print('last tick: %s' % str(new_tick_time))
                    print('Completed run_simulation() in %i seconds at %i ticks/second' %
                          (elapsed, int(coinbase_counter/elapsed)))
                    print('***Looped through %i rows***' % coinbase_counter)
                    # end the loop
                    break

        return snapshot_list

    @staticmethod
    def export_snapshots_to_csv(orderbook_snapshot_history):
        filepath = './orderbook_snapshot_history.csv'

        columns = list()
        columns.append('system_time')
        for exchange in ['coinbase', 'bitfinex']:
            for side in ['bid', 'ask']:
                for feature in ['notional', 'distance']:
                    for level in range(MAX_BOOK_ROWS):
                        columns.append(('%s-%s-%s-%i' % (exchange, side, feature, level)))

        panda_orderbook_snapshot_history = pd.DataFrame(orderbook_snapshot_history, columns=columns)
        panda_orderbook_snapshot_history = panda_orderbook_snapshot_history.dropna(axis=0, inplace=False)
        panda_orderbook_snapshot_history.to_csv(filepath)
        print('Exported order book snapshots to csv.')


if __name__ == '__main__':
    print('Testing is commencing...')

    query = {
        'ccy': ['LTC-USD', 'tLTCUSD'],
        'start_date': 20180926,
        'end_date': 20180927
    }

    coinbaseOrderBook = CoinbaseOrderBook(query['ccy'][0])
    bitfinexOrderBook = BitfinexOrderBook(query['ccy'][1])

    sim = Simulator()
    tick_history = sim.get_tick_history(query)
    orderbook_snapshot_history = sim.get_orderbook_snapshot_history(coinbaseOrderBook, bitfinexOrderBook, tick_history)

    # export to CSV to verify if order book reconstruction is accurate/good
    # Note: this is only to show that the functionality works and
    #       should be fed into an Environment for reinforcement learning.
    sim.export_snapshots_to_csv(orderbook_snapshot_history)

    print('DONE. EXITING %s' % __name__)
