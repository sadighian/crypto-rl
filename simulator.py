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
from sklearn.preprocessing import StandardScaler


class Simulator(object):

    def __init__(self):
        try:
            print('Attempting to connect to Arctic...')
            self.scaler = StandardScaler()
            self.arctic = Arctic(MONGO_ENDPOINT)
            self.arctic.initialize_library(ARCTIC_NAME, lib_type=TICK_STORE)
            self.library = self.arctic[ARCTIC_NAME]
            self.number_of_workers = cpu_count()
            self.queue = Queue(maxsize=self.number_of_workers)
            self.return_queue = Queue(maxsize=self.number_of_workers)
            self.workers = [Process(name='Process-%i' % num,
                                    args=(self.queue, self.return_queue),
                                    target=self._do_work) for num in range(self.number_of_workers)]
            print('Connected to Arctic.')
        except Exception as ex:
            self.arctic, self.library, self.workers = None, None, None
            print('Unable to connect to Arctic database')
            print(ex)

    def get_tick_history(self, query):
        """
        Function to query the Arctic Tick Store and...
        1.  Return the specified historical data for a given set of securities
            over a specified amount of time
        2.  Convert the data returned from the query from a panda to a list of dicts
            and while doing so, allocate the work across all available CPU cores

        :param query: (dict) of the query parameters
            - ccy: list of symbols
            - startDate: int YYYYMMDD
            - endDate: int YYYYMMDD
        :return: list of dicts, where each dict is a tick that was recorded
        """
        start_time = dt.now(TIMEZONE)

        assert RECORD_DATA is False

        tick_history_dict = SortedDict()

        cursor = self._query_arctic(**query)

        if cursor is None:
            print('\nNothing returned from Arctic for the query: %s\n...Exiting...' % str(query))
            return

        self._split_cursor_and_add_to_queue(cursor)

        del cursor
        counter = 0

        [worker.start() for worker in self.workers]

        while True:
            cpu, list_of_dicts = self.return_queue.get()
            tick_history_dict[cpu] = list_of_dicts
            counter += 1

            if counter == self.number_of_workers:
                tick_history_list = list(sum(tick_history_dict.values()[:], []))
                del tick_history_dict
                break

        [worker.join() for worker in self.workers]

        elapsed = (dt.now(TIMEZONE) - start_time).seconds
        print('***\nCompleted get_tick_history() in %i seconds\n***' % elapsed)

        return tick_history_list

    def _query_arctic(self, ccy, start_date, end_date):
        start_time = dt.now(TIMEZONE)

        if self.library is None:
            print('exiting from Simulator... no database to query')
            return None

        try:
            print('\nGetting %s tick data from Arctic Tick Store...' % ccy)
            cursor = self.library.read(symbol=ccy, date_range=DateRange(start_date, end_date))

            # filter ticks for the first LOAD_BOOK message (starting point for order book reconstruction)
            min_datetime = cursor.loc[cursor.type == 'load_book'].index[0]
            cursor = cursor.loc[cursor.index >= min_datetime].copy()

            elapsed = (dt.now(TIMEZONE) - start_time).seconds
            print('Completed querying %i %s records in %i seconds' % (cursor.shape[0], ccy, elapsed))

        except Exception as ex:
            cursor = None
            print('Simulator._query_arctic() thew an exception: \n%s' % str(ex))

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
        """
        Function used for parallel processing
        :param queue: queue for dequeing tasks
        :param return_queue: queue for enqueing completed tasks
        :return: void
        """
        start_time = dt.now(TIMEZONE)

        cpu, panda = queue.get()
        results = panda.to_dict('records')
        return_queue.put((cpu, results))

        elapsed = (dt.now(TIMEZONE) - start_time).seconds
        print('Completed. Process-%i dataframe.to_dict() in %i seconds' % (cpu, elapsed))

    @staticmethod
    def get_orderbook_snapshot_history(coinbase_order_book, bitfinex_order_book, tick_history):
        """
        Function to replay historical market data and generate
        the features used for reinforcement learning & training
        :param coinbase_order_book: Coinbase Orderbook class
        :param bitfinex_order_book: Bitfinex Orderbook class
        :param tick_history: (list of dicts) historical tick data
        :return: (list of arrays) snapshots of limit order books using a stationary feature set
        """
        start_time = dt.now(TIMEZONE)

        coinbase_tick_counter = 0
        snapshot_list = list()
        last_snapshot_time = None
        loop_length = len(tick_history)

        print('Starting get_orderbook_snapshot_history() with %i ticks for %s and %s' %
              (loop_length, coinbase_order_book.sym, bitfinex_order_book.sym))

        for idx, tick in enumerate(tick_history):
            # determine if incoming tick is from coinbase or bitfinex
            coinbase = True if tick['product_id'] == coinbase_order_book.sym else False

            if 'type' not in tick:
                # filter out bad ticks
                continue

            if tick['type'] in ['load_book', 'book_loaded', 'preload']:
                # flag for a order book reset
                if coinbase:
                    coinbase_order_book.new_tick(tick)
                else:
                    bitfinex_order_book.new_tick(tick)
                # skip to next loop
                continue

            if coinbase:
                if coinbase_order_book.done_warming_up():
                    new_tick_time = parse(tick['time'])  # timestamp for incoming tick
                    coinbase_tick_counter += 1
                    coinbase_order_book.new_tick(tick)

                if coinbase_tick_counter == 1:
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
                        midpoint_delta = coinbase_order_book.midpoint - bitfinex_order_book.midpoint
                        for _ in range(multiple):
                            snapshot_list.append(list(np.hstack((new_tick_time,  # tick time
                                                                 coinbase_order_book.midpoint,  # midpoint price
                                                                 midpoint_delta,  # price delta between exchanges
                                                                 coinbase_order_book_snapshot,
                                                                 bitfinex_order_book_snapshot))))
                            last_snapshot_time += timedelta(milliseconds=250)
                    else:
                        last_snapshot_time += timedelta(milliseconds=250)

            elif bitfinex_order_book.done_warming_up():
                bitfinex_order_book.new_tick(tick)

            if idx % 150000 == 0:
                elapsed = (dt.now(TIMEZONE) - start_time).seconds
                print('...completed %i loops in %i seconds' % (idx, elapsed))

        elapsed = (dt.now(TIMEZONE) - start_time).seconds
        print('last tick: %s' % str(new_tick_time))
        print('Completed run_simulation() with %i ticks in %i seconds at %i ticks/second' %
              (loop_length, elapsed, int(loop_length/elapsed)))
        print('***\nLooped through %i ticks and %i coinbase ticks\n***' % (loop_length, coinbase_tick_counter))

        return snapshot_list

    @staticmethod
    def get_feature_labels(include_system_time=True, lags=0):
        columns = list()

        if include_system_time:
            columns.append('system_time')

        if lags > 0:
            for lag in range(lags+1):
                columns.append('coinbase_midpoint---%i' % lag)
                columns.append('midpoint_delta---%i' % lag)
                for exchange in ['coinbase', 'bitfinex']:
                    for side in ['bid', 'ask']:
                        for feature in ['notional', 'distance']:
                            for level in range(MAX_BOOK_ROWS):
                                columns.append(('%s-%s-%s-%i---%i' % (exchange, side, feature, level, lag)))
                    for trade_side in ['buys', 'sells']:
                        columns.append('%s-%s---%i' % (exchange, trade_side, lag))
        else:
            columns.append('coinbase_midpoint')
            columns.append('midpoint_delta')
            for exchange in ['coinbase', 'bitfinex']:
                for side in ['bid', 'ask']:
                    for feature in ['notional', 'distance']:
                        for level in range(MAX_BOOK_ROWS):
                            columns.append(('%s-%s-%s-%i' % (exchange, side, feature, level)))
                for trade_side in ['buys', 'sells']:
                    columns.append('%s-%s' % (exchange, trade_side))

        return columns

    @staticmethod
    def export_snapshots_to_csv(columns, orderbook_snapshot_history):
        start_time = dt.now(TIMEZONE)

        panda_orderbook_snapshot_history = pd.DataFrame(orderbook_snapshot_history, columns=columns)
        panda_orderbook_snapshot_history = panda_orderbook_snapshot_history.dropna(axis=0, inplace=False)
        panda_orderbook_snapshot_history.to_csv('./orderbook_snapshot_history.csv')

        elapsed = (dt.now(TIMEZONE) - start_time).seconds
        print('Exported order book snapshots to csv in %i seconds' % elapsed)

    def get_orderbook_snapshot_history_normalized(self, query, lags):
        start_time = dt.now(tz=TIMEZONE)

        data = self.get_orderbook_snapshot_history(CoinbaseOrderBook(query['ccy'][0]),  # order book
                                                   BitfinexOrderBook(query['ccy'][1]),  # order book
                                                   self.get_tick_history(query))  # historical data

        data = pd.DataFrame(data, columns=self.get_feature_labels(True))
        data = data.dropna(axis=0)

        data['day'] = data['system_time'].apply(lambda x: x.day)
        dates = data['day'].unique()

        assert len(dates) > 1  # make sure at least 2 days of data was loaded

        data_to_fit = data.loc[data.day == dates[0], :].copy()
        data_to_fit = data_to_fit.drop(['system_time', 'day'], axis=1)
        self.scaler.fit(data_to_fit)

        if lags > 0:
            data = data.loc[data.day.shift(-lags) == dates[1], :]  # include lags in dataset
        else:
            data = data.loc[data.day == dates[1], :]  # do not include lags in dataset

        data = data.drop(['system_time', 'day'], axis=1)

        data_transformed = self.scaler.transform(data)

        elapsed = (dt.now(tz=TIMEZONE) - start_time).seconds
        print('***\nSimulator.get_orderbook_snapshot_history_normalized() executed in %i seconds\n***' % elapsed)

        return data_transformed

    def get_env_data(self, query, lags=0):
        normalized_data = self.get_orderbook_snapshot_history_normalized(query=query, lags=lags)
        data = pd.DataFrame(normalized_data, columns=self.get_feature_labels(False))

        # add lags to feature set
        if lags > 0:
            _columns = data.columns.tolist()
            for lag in range(1, lags+1):
                for col in _columns:
                    data[('%s-%i' % (col, lag))] = data[col].shift(lag)

        data.columns = self.get_feature_labels(False, lags=lags)
        data = data.dropna(axis=0)

        # self.export_env_data_to_csv(data_transformed=data)
        return data

    @staticmethod
    def export_env_data_to_csv(env_data):
        start_time = dt.now(TIMEZONE)

        env_data.to_csv('./env_data_history.csv')

        elapsed = (dt.now(TIMEZONE) - start_time).seconds
        print('Exported env_data to csv in %i seconds' % elapsed)


def test_get_orderbook_snapshot_history():
    start_time = dt.now(TIMEZONE)

    query = {
        'ccy': ['BCH-USD', 'tBCHUSD'],
        'start_date': 20181110,
        'end_date': 20181112
    }

    coinbaseOrderBook = CoinbaseOrderBook(query['ccy'][0])
    bitfinexOrderBook = BitfinexOrderBook(query['ccy'][1])

    sim = Simulator()
    tick_history = sim.get_tick_history(query)

    if tick_history is None:
        print('Exiting due to no data being available.')
        return

    orderbook_snapshot_history = sim.get_orderbook_snapshot_history(coinbaseOrderBook, bitfinexOrderBook, tick_history)

    # Export to CSV to verify if order book reconstruction is accurate/good
    # NOTE: this is only to show that the functionality works and
    #       should be fed into an Environment for reinforcement learning.
    sim.export_snapshots_to_csv(sim.get_feature_labels(True), orderbook_snapshot_history)

    elapsed = (dt.now(TIMEZONE) - start_time).seconds
    print('Completed %s in %i seconds' % (__name__, elapsed))
    print('DONE. EXITING %s' % __name__)


def test_get_env_data():
    start_time = dt.now(TIMEZONE)

    query = {
        'ccy': ['BCH-USD', 'tBCHUSD'],
        'start_date': 20181110,
        'end_date': 20181112
    }

    lags = 4*60

    sim = Simulator()
    env_data = sim.get_env_data(query=query, lags=lags)

    sim.export_env_data_to_csv(env_data)

    elapsed = (dt.now(TIMEZONE) - start_time).seconds
    print('Completed %s in %i seconds' % (__name__, elapsed))
    print('DONE. EXITING %s' % __name__)


if __name__ == '__main__':
    """
    Entry point of simulation application
    """
    test_get_orderbook_snapshot_history()
    # test_get_env_data()
