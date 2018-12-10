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
from sklearn.preprocessing import MinMaxScaler


class Simulator(object):

    def __init__(self):
        try:
            print('Attempting to connect to Arctic...')
            self.scaler = MinMaxScaler()
            self.arctic = Arctic(MONGO_ENDPOINT)
            self.arctic.initialize_library(ARCTIC_NAME, lib_type=TICK_STORE)
            self.library = self.arctic[ARCTIC_NAME]
            self.reference_data_old_names = ['system_time', 'day', 'coinbase_midpoint']
            self.reference_data_new_names = ['t', 'd', 'm']
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

    def reset_processes(self):
        self.queue = Queue(maxsize=self.number_of_workers)
        self.return_queue = Queue(maxsize=self.number_of_workers)
        self.workers = [Process(name='Process-%i' % num,
                                args=(self.queue, self.return_queue),
                                target=self._do_work) for num in range(self.number_of_workers)]
        print('Simulator.reset_processes()')

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

        [worker.start() for worker in self.workers]

        counter = 0

        while True:
            cpu, list_of_dicts = self.return_queue.get()
            tick_history_dict[cpu] = list_of_dicts
            counter += 1

            if counter == self.number_of_workers:
                tick_history_list = list(sum(tick_history_dict.values()[:], []))
                del tick_history_dict
                break

        [worker.join() for worker in self.workers]
        del cursor

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
    def get_feature_labels(include_system_time=True, lags=0):
        """
        Function to create the features' labels
        :param include_system_time: True/False (False removes the system_time column)
        :param lags: Number of lags to include
        :return:
        """
        columns = list()

        if include_system_time:
            columns.append('system_time')

        for lag in range(lags + 1):
            suffix = '' if lag == 0 else '_%i' % lag
            columns.append('coinbase_midpoint%s' % suffix)
            columns.append('midpoint_delta%s' % suffix)
            for exchange in ['coinbase', 'bitfinex']:
                for side in ['bid', 'ask']:
                    for feature in ['notional', 'distance']:
                        for level in range(MAX_BOOK_ROWS):
                            columns.append(('%s-%s-%s-%i%s' % (exchange, side, feature, level, suffix)))
                for trade_side in ['buys', 'sells']:
                    columns.append('%s-%s%s' % (exchange, trade_side, suffix))

        return columns

    def get_orderbook_snapshot_history(self, coinbase_order_book, bitfinex_order_book, tick_history):
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

            if coinbase:  # incoming tick is from Coinbase exchange
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

                # calculate the amount of time between the incoming tick and tick received before that
                diff = (new_tick_time - last_snapshot_time).microseconds
                multiple = diff // 250000  # 250000 is 250 milliseconds, or 4x a second

                if multiple >= 1:  # if there is a pause in incoming data, continue to create order book snapshots
                    for _ in range(multiple):
                        if coinbase_order_book.done_warming_up() & bitfinex_order_book.done_warming_up():
                            coinbase_order_book_snapshot = coinbase_order_book.render_book()
                            bitfinex_order_book_snapshot = bitfinex_order_book.render_book()
                            midpoint_delta = coinbase_order_book.midpoint - bitfinex_order_book.midpoint
                            snapshot_list.append(list(np.hstack((new_tick_time,  # tick time
                                                                 coinbase_order_book.midpoint,  # midpoint price
                                                                 midpoint_delta,  # price delta between exchanges
                                                                 coinbase_order_book_snapshot,
                                                                 bitfinex_order_book_snapshot))))  # longs/shorts
                            last_snapshot_time += timedelta(milliseconds=250)

                        else:
                            last_snapshot_time += timedelta(milliseconds=250)

            # incoming tick is from Bitfinex exchange
            elif bitfinex_order_book.done_warming_up():
                bitfinex_order_book.new_tick(tick)

            # periodically print number of steps completed
            if idx % 250000 == 0:
                elapsed = (dt.now(TIMEZONE) - start_time).seconds
                print('...completed %i loops in %i seconds' % (idx, elapsed))

        elapsed = (dt.now(TIMEZONE) - start_time).seconds
        print('Completed run_simulation() with %i ticks in %i seconds at %i ticks/second' %
              (loop_length, elapsed, loop_length//elapsed))

        orderbook_snapshot_history = pd.DataFrame(snapshot_list, columns=self.get_feature_labels())
        orderbook_snapshot_history = orderbook_snapshot_history.dropna(axis=0)

        return orderbook_snapshot_history

    # def normalize_orderbook_snapshot_history(self, orderbook_snapshot_history):
    #     start_time = dt.now(tz=TIMEZONE)
    #
    #     data = orderbook_snapshot_history.copy()
    #
    #     data['day'] = data['system_time'].apply(lambda x: x.day)
    #     dates = data['day'].unique()
    #     assert len(dates) > 1  # make sure at least 2 days of data was loaded
    #
    #     data_to_fit = data.loc[data.day == dates[0], :].copy()
    #     data_to_fit = data_to_fit.drop(['system_time', 'day'], axis=1)
    #     self.scaler.fit(data_to_fit)
    #
    #     # data = data.drop(['system_time', 'day'], axis=1)
    #
    #     column_names = self.get_feature_labels(False)
    #
    #     normalized_data = pd.DataFrame(self.scaler.transform(data[column_names]),
    #                                    index=data.index,
    #                                    columns=column_names)
    #
    #     reference_data = data[self.reference_data_old_names]
    #     reference_data.columns = self.reference_data_new_names
    #
    #     normalized_data = pd.concat((normalized_data, reference_data), axis=1)
    #
    #     elapsed = (dt.now(tz=TIMEZONE) - start_time).seconds
    #     print('***\nSimulator.get_orderbook_snapshot_history_normalized() executed in %i seconds\n***' % elapsed)
    #
    #     return normalized_data

    # def append_lags(self, data, lags=0):
    #     start_time = dt.now()
    #
    #     temp = data.copy()
    #
    #     # add lags to feature set
    #     if lags > 0:
    #         columns = self.get_feature_labels(False)
    #         for lag in range(1, lags + 1):
    #             for col in columns:
    #                 temp[('%s_%i' % (col, lag))] = temp[col].shift(lag)
    #
    #     temp = temp.dropna(axis=0)
    #
    #     elapsed = (dt.now() - start_time).seconds
    #     print('***\nSimulator.append_lags() executed in %i seconds\n***' % elapsed)
    #     return temp

    def fit_scaler(self, orderbook_snapshot_history):
        start_time = dt.now(tz=TIMEZONE)

        self.scaler.fit(orderbook_snapshot_history)

        elapsed = (dt.now(tz=TIMEZONE) - start_time).seconds
        print('***\nSimulator._fit_scaler() executed in %i seconds\n***' % elapsed)

    def scale_state(self, _next_state):
        return self.scaler.transform(_next_state.reshape(1, -1)).reshape(_next_state.shape)
              #self.scaler.transform(_next_state.reshape(1, -1)).reshape(_next_state.shape)

    def get_scaler(self):
        return self.scaler

    def _get_env_states(self, query):
        start_time = dt.now(TIMEZONE)

        tick_history = self.get_tick_history(query)

        coinbaseOrderBook = CoinbaseOrderBook(query['ccy'][0])
        bitfinexOrderBook = BitfinexOrderBook(query['ccy'][1])

        orderbook_snapshot_history = self.get_orderbook_snapshot_history(coinbaseOrderBook,
                                                                         bitfinexOrderBook,
                                                                         tick_history)
        elapsed = (dt.now(TIMEZONE) - start_time).seconds
        print('Sim.get_env_states() executed in %i seconds' % elapsed)
        return orderbook_snapshot_history

    def query_env_states(self, query):
        start_time = dt.now(TIMEZONE)

        # fetch the previous day's data to use for normalizing the data above
        query['start_date'] -= 1
        query['end_date'] -= 1
        history_for_fitting = self._get_env_states(query=query)
        self.fit_scaler(orderbook_snapshot_history=history_for_fitting.drop(['system_time'], axis=1))
        del history_for_fitting
        self.reset_processes()

        # fetch data for the environment (needs to be normalized within trading_gym)
        query['start_date'] += 1
        query['end_date'] += 1
        history = self._get_env_states(query=query)

        elapsed = (dt.now(TIMEZONE) - start_time).seconds
        print('Sim.load_env_states() executed in %i seconds' % elapsed)
        return history

    def load_env_states(self, fitting_filepath, env_filepath):
        try:
            orderbook_data = pd.read_csv(fitting_filepath)
            del orderbook_data[orderbook_data.columns[0]]
            del orderbook_data['system_time']

            self.fit_scaler(orderbook_snapshot_history=orderbook_data)

            orderbook_data = pd.read_csv(env_filepath)
            del orderbook_data[orderbook_data.columns[0]]
            del orderbook_data['system_time']

            # orderbook_data['system_time'] = orderbook_data['system_time'].apply(lambda x: pd.to_datetime(x))
            print('Simulator.load_env_states() env data loaded')

            return orderbook_data
        except Exception as e:
            print('Simulator.load_env_states() unable to load data')
            print(e)
            return None

    @staticmethod
    def export_to_csv(data, filename):
        start_time = dt.now(TIMEZONE)

        data.to_csv(filename)

        elapsed = (dt.now(TIMEZONE) - start_time).seconds
        print('Exported data to csv in %i seconds' % elapsed)
