from datetime import datetime as dt
from datetime import timedelta
from arctic import Arctic, TICK_STORE
from arctic.date import DateRange
from data_recorder.coinbase_connector.coinbase_orderbook import CoinbaseOrderBook
from data_recorder.bitfinex_connector.bitfinex_orderbook import BitfinexOrderBook
from configurations.configs import TIMEZONE, MONGO_ENDPOINT, ARCTIC_NAME, \
    RECORD_DATA, MAX_BOOK_ROWS
from dateutil.parser import parse
import numpy as np
import pandas as pd
import os


class Simulator(object):

    def __init__(self, use_arctic=False):
        """

        :param use_arctic: If True, Simulator creates a connection to Arctic,
                            Otherwise, no connection is attempted
        """
        self._avg = None
        self._std = None
        self.cwd = os.path.dirname(os.path.realpath(__file__))
        self.z_score = lambda x: (x - self._avg) / self._std
        try:
            if use_arctic:
                print('Attempting to connect to Arctic...')
                self.arctic = Arctic(MONGO_ENDPOINT)
                self.arctic.initialize_library(ARCTIC_NAME, lib_type=TICK_STORE)
                self.library = self.arctic[ARCTIC_NAME]
                print('Connected to Arctic.')
            else:
                print('Not connecting to Arctic')
                self.arctic, self.library = None, None
        except Exception as ex:
            self.arctic, self.library = None, None
            print('Unable to connect to Arctic database')
            print(ex)

    def __str__(self):
        return 'Simulator() connection={}, library={}, avg={}, std={}' \
            .format(self.arctic, self.library, self._avg, self._std)

    def scale_state(self, next_state):
        return (next_state - self._avg) / self._std

    def reset(self):
        """
        Sets averages and standard deviations to NONE
        :return: void
        """
        self._avg = None
        self._std = None

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
        start_time = dt.now(tz=TIMEZONE)

        assert RECORD_DATA is False
        cursor = self._query_arctic(**query)
        if cursor is None:
            print('\nNothing returned from Arctic for the query: %s\n...Exiting...'
                  % str(query))
            return

        elapsed = (dt.now(tz=TIMEZONE) - start_time).seconds
        print('***\nCompleted get_tick_history() in %i seconds\n***' % elapsed)

        return cursor

    def _query_arctic(self, ccy, start_date, end_date):
        start_time = dt.now(tz=TIMEZONE)

        if self.library is None:
            print('exiting from Simulator... no database to query')
            return None

        try:
            print('\nGetting {} tick data from Arctic Tick Store...'.format(ccy))
            cursor = self.library.read(symbol=ccy,
                                       date_range=DateRange(start_date, end_date))

            # filter ticks for the first LOAD_BOOK message
            #   (starting point for order book reconstruction)
            # min_datetime = cursor.loc[cursor.type == 'load_book'].index[0]
            dates = np.unique(cursor.loc[cursor.type == 'load_book'].index.date)
            start_index = cursor.loc[((cursor.index.date == dates[0]) &
                                      (cursor.type == 'load_book'))].index[-1]
            # cursor = cursor.loc[cursor.index >= min_datetime]
            cursor = cursor.loc[cursor.index >= start_index]

            elapsed = (dt.now(tz=TIMEZONE) - start_time).seconds
            print('Completed querying %i %s records in %i seconds' %
                  (cursor.shape[0], ccy, elapsed))

        except Exception as ex:
            cursor = None
            print('Simulator._query_arctic() thew an exception: \n%s' % str(ex))

        return cursor

    @staticmethod
    def get_feature_labels(include_system_time=True, include_bitfinex=True):
        """
        Function to create the features' labels
        :param include_bitfinex: (boolean) If TRUE, Bitfinex's LOB data
                is included in the dataset, in addition to Coinbase-Pro
        :param include_system_time: True/False
                (False removes the system_time column)
        :return:
        """
        columns = list()

        if include_system_time:
            columns.append('system_time')

        columns.append('coinbase_midpoint')

        exchanges = ['coinbase']
        if include_bitfinex:
            columns.append('midpoint_delta')
            exchanges.append('bitfinex')

        for exchange in exchanges:
            for side in ['bid', 'ask']:
                for feature in ['notional', 'distance']:
                    for level in range(MAX_BOOK_ROWS):
                        columns.append(('%s-%s-%s-%i' %
                                        (exchange, side, feature, level)))
            for trade_side in ['buys', 'sells']:
                columns.append('%s-%s' % (exchange, trade_side))

        return columns

    def export_to_csv(self, data, filename='BTC-USD_2019-01-01', compress=True):
        """
        Export data within a Panda dataframe to a csv
        :param data: (panda.DataFrame) historical tick data
        :param filename: CCY_YYYY-MM-DD
        :param compress: Default True. If True, compress with xz
        :return: void
        """
        start_time = dt.now(tz=TIMEZONE)

        sub_folder = '{}/data_exports/{}'.format(self.cwd, filename)

        if compress:
            sub_folder += '.xz'
            data.to_csv(path_or_buf=sub_folder, index=False, compression='xz')
        else:
            sub_folder += '.csv'
            data.to_csv(path_or_buf=sub_folder, index=False)

        elapsed = (dt.now(tz=TIMEZONE) - start_time).seconds
        print('Exported %s with %i rows in %i seconds' %
              (sub_folder, data.shape[0], elapsed))

    @staticmethod
    def import_csv(filename='data.xz'):
        """
        Import an historical tick file created from the
        export_to_csv() function
        :param filename: Full file path including filename
        :return: (panda.DataFrame) historical limit order book data
        """
        start_time = dt.now(tz=TIMEZONE)

        if 'xz' in filename:
            data = pd.read_csv(filepath_or_buffer=filename, index_col=0,
                               compression='xz', engine='c')
        elif 'csv' in filename:
            data = pd.read_csv(filepath_or_buffer=filename, index_col=0,
                               engine='c')
        else:
            print('Error: file must be a csv or xz')
            data = None

        elapsed = (dt.now(tz=TIMEZONE) - start_time).seconds
        print('Imported %s from a csv in %i seconds' % (filename[-21:], elapsed))
        return data

    def fit_scaler(self, orderbook_snapshot_history):
        """
        Fit scalers for z-score using previous day's data
        :param orderbook_snapshot_history: Limit order book data
                from the previous day
        :return: void
        """
        self._avg = np.mean(orderbook_snapshot_history, axis=0)
        self._std = np.std(orderbook_snapshot_history, axis=0)

    def extract_features(self, query):
        """
        Create and export limit order book data to csv. This function
        exports multiple days of data and ensures each day starts and
        ends exactly on time.
        :param query: (dict) ccy= symy, daterange=YYYYMMDD,YYYYMMDD
        :return: void
        """
        start_time = dt.now(tz=TIMEZONE)

        order_book_data = self.get_orderbook_snapshot_history(query=query)
        if order_book_data is not None:
            dates = order_book_data['system_time'].dt.date.unique()
            print('dates: {}'.format(dates))
            for date in dates[1:]:
                tmp = order_book_data.loc[order_book_data['system_time'].dt.date
                                          == date]
                self.export_to_csv(tmp,
                                   filename='{}_{}'.format(query['ccy'][0], date),
                                   compress=False)

        elapsed = (dt.now(tz=TIMEZONE) - start_time).seconds
        print('***\nSimulator.extract_features() executed in %i seconds\n***'
              % elapsed)

    def get_orderbook_snapshot_history(self, query):
        """
        Function to replay historical market data and generate
        the features used for reinforcement learning & training.

        NOTE:
        The query can either be a single Coinbase CCY, or both Coinbase and Bitfinex,
        but it cannot be only a Biftinex CCY. Later releases of this repo will
        support Bitfinex only orderbook reconstruction.

        :param query: (dict) query for finding tick history in Arctic TickStore
        :return: (list of arrays) snapshots of limit order books using a
                stationary feature set
        """
        tick_history = self.get_tick_history(query=query)
        loop_length = tick_history.shape[0]

        coinbase_tick_counter = 0
        snapshot_list = list()
        last_snapshot_time = None

        symbols = query['ccy']
        print('querying {}'.format(symbols))

        include_bitfinex = len(symbols) > 1
        if include_bitfinex:
            print('\n\nIncluding Bitfinex data in feature set.\n\n')

        coinbase_order_book = CoinbaseOrderBook(symbols[0])
        bitfinex_order_book = BitfinexOrderBook(symbols[1]) if include_bitfinex \
            else None

        start_time = dt.now(TIMEZONE)
        print('Starting get_orderbook_snapshot_history() loop with %i ticks for %s'
              % (loop_length, query['ccy']))

        for idx, tx in enumerate(tick_history.itertuples()):

            tick = tx._asdict()

            # determine if incoming tick is from coinbase or bitfinex
            coinbase = True if tick['product_id'] == coinbase_order_book.sym else \
                False

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
                    new_tick_time = parse(tick.get('time'))
                    # timestamp for incoming tick
                    if new_tick_time is None:
                        print('No tick time: {}'.format(tick))
                        continue

                    coinbase_tick_counter += 1
                    coinbase_order_book.new_tick(tick)

                if coinbase_tick_counter == 1:
                    # start tracking snapshot timestamps
                    #   and keep in mind that snapshots are tethered to
                    #   coinbase timestamps
                    last_snapshot_time = new_tick_time
                    print('%s first tick: %s | Sequence: %i' %
                          (coinbase_order_book.sym, str(new_tick_time),
                           coinbase_order_book.sequence))
                    # skip to next loop
                    continue

                # calculate the amount of time between the incoming
                #   tick and tick received before that
                diff = (new_tick_time - last_snapshot_time).microseconds

                # multiple = diff // 250000
                multiple = diff // 500000  # 500000 is 500 milliseconds,
                #   or 2x a second

                # if there is a pause in incoming data,
                #   continue to create order book snapshots
                if multiple >= 1:
                    # check to include Bitfinex data in features
                    if include_bitfinex:
                        for _ in range(multiple):
                            if coinbase_order_book.done_warming_up() & \
                                    bitfinex_order_book.done_warming_up():
                                coinbase_order_book_snapshot = \
                                    coinbase_order_book.render_book()
                                bitfinex_order_book_snapshot = \
                                    bitfinex_order_book.render_book()
                                midpoint_delta = coinbase_order_book.midpoint - \
                                                 bitfinex_order_book.midpoint
                                snapshot_list.append(list(np.hstack((
                                    new_tick_time,  # tick time
                                    coinbase_order_book.midpoint,  # midpoint price
                                    midpoint_delta,  # price delta between exchanges
                                    coinbase_order_book_snapshot,
                                    bitfinex_order_book_snapshot))))  # longs/shorts
                                last_snapshot_time += timedelta(milliseconds=500)

                            else:
                                last_snapshot_time += timedelta(milliseconds=500)
                    else:  # do not include bitfinex
                        for _ in range(multiple):
                            if coinbase_order_book.done_warming_up():
                                coinbase_order_book_snapshot = \
                                    coinbase_order_book.render_book()
                                snapshot_list.append(
                                    list(np.hstack((new_tick_time,  # tick time
                                                    coinbase_order_book.midpoint,
                                                    coinbase_order_book_snapshot))))
                                last_snapshot_time += timedelta(milliseconds=500)

                            else:
                                last_snapshot_time += timedelta(milliseconds=500)

            # incoming tick is from Bitfinex exchange
            elif include_bitfinex & bitfinex_order_book.done_warming_up():
                bitfinex_order_book.new_tick(tick)

            # periodically print number of steps completed
            if idx % 250000 == 0:
                elapsed = (dt.now(TIMEZONE) - start_time).seconds
                print('...completed %i loops in %i seconds' % (idx, elapsed))

        elapsed = (dt.now(TIMEZONE) - start_time).seconds
        print('Completed run_simulation() with %i ticks in %i seconds '
              'at %i ticks/second'
              % (loop_length, elapsed, loop_length//elapsed))

        orderbook_snapshot_history = pd.DataFrame(
            snapshot_list,
            columns=self.get_feature_labels(
                include_system_time=True,
                include_bitfinex=include_bitfinex))
        orderbook_snapshot_history = orderbook_snapshot_history.dropna(axis=0)

        return orderbook_snapshot_history
