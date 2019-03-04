from datetime import datetime as dt
from datetime import timedelta
from arctic import Arctic, TICK_STORE
from arctic.date import DateRange
from coinbase_connector.coinbase_orderbook import CoinbaseOrderBook
from bitfinex_connector.bitfinex_orderbook import BitfinexOrderBook
from configurations.configs import TIMEZONE, MONGO_ENDPOINT, ARCTIC_NAME, RECORD_DATA, MAX_BOOK_ROWS
from dateutil.parser import parse
import numpy as np
import pandas as pd
import os


class Simulator(object):

    def __init__(self, use_arctic=False):
        self._avg = None
        self._std = None
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

    def reset(self):
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
        start_time = dt.now(TIMEZONE)

        assert RECORD_DATA is False
        cursor = self._query_arctic(**query)
        if cursor is None:
            print('\nNothing returned from Arctic for the query: %s\n...Exiting...' % str(query))
            return

        elapsed = (dt.now(TIMEZONE) - start_time).seconds
        print('***\nCompleted get_tick_history() in %i seconds\n***' % elapsed)

        return cursor

    def _query_arctic(self, ccy, start_date, end_date):
        start_time = dt.now(TIMEZONE)

        if self.library is None:
            print('exiting from Simulator... no database to query')
            return None

        try:
            print('\nGetting %s tick data from Arctic Tick Store...' % ccy)
            cursor = self.library.read(symbol=ccy, date_range=DateRange(start_date, end_date))

            # filter ticks for the first LOAD_BOOK message (starting point for order book reconstruction)
            # min_datetime = cursor.loc[cursor.type == 'load_book'].index[0]
            dates = np.unique(cursor.loc[cursor.type == 'load_book'].index.date)
            start_index = cursor.loc[((cursor.index.date == dates[0]) &
                                      (cursor.type == 'load_book'))].index[-1]
            # cursor = cursor.loc[cursor.index >= min_datetime]
            cursor = cursor.loc[cursor.index >= start_index]

            elapsed = (dt.now(TIMEZONE) - start_time).seconds
            print('Completed querying %i %s records in %i seconds' % (cursor.shape[0], ccy, elapsed))

        except Exception as ex:
            cursor = None
            print('Simulator._query_arctic() thew an exception: \n%s' % str(ex))

        return cursor

    @staticmethod
    def get_feature_labels(include_system_time=True, include_bitfinex=True):
        """
        Function to create the features' labels
        :param include_system_time: True/False (False removes the system_time column)
        :param lags: Number of lags to include
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
                        columns.append(('%s-%s-%s-%i' % (exchange, side, feature, level)))
            for trade_side in ['buys', 'sells']:
                columns.append('%s-%s' % (exchange, trade_side))

        return columns

    @staticmethod
    def export_to_csv(data, filename='data', compress=True):
        start_time = dt.now(tz=TIMEZONE)

        cwd = os.getcwd() + '/trading_gym/data_exports/'

        if compress:
            subfolder = cwd + '{}.xz'.format(filename)
            data.to_csv(path_or_buf=subfolder, index=False, compression='xz')
        else:
            subfolder = cwd + '{}.csv'.format(filename)
            data.to_csv(path_or_buf=subfolder, index=False)

        elapsed = (dt.now(tz=TIMEZONE) - start_time).seconds
        print('Exported %s to csv in %i seconds' % (filename, elapsed))

    @staticmethod
    def import_csv(filename='data.xz'):
        start_time = dt.now(tz=TIMEZONE)

        if 'xz' in filename:
            data = pd.read_csv(filepath_or_buffer=filename, index_col=0, compression='xz', engine='c')
        elif 'csv' in filename:
            data = pd.read_csv(filepath_or_buffer=filename, index_col=0, engine='c')
        else:
            print('Error: file must be a csv or xz')
            data = None

        elapsed = (dt.now(tz=TIMEZONE) - start_time).seconds
        print('Imported %s from a csv in %i seconds' % (filename[-21:], elapsed))
        return data

    def fit_scaler(self, orderbook_snapshot_history):
        self._avg = np.mean(orderbook_snapshot_history, axis=0)
        self._std = np.std(orderbook_snapshot_history, axis=0)

    def scale_state(self, _next_state):
        return (_next_state - self._avg) / self._std

    def extract_features(self, query):
        start_time = dt.now(tz=TIMEZONE)

        order_book_data = self.get_orderbook_snapshot_history(query=query)
        if order_book_data is not None:
            dates = order_book_data['system_time'].dt.date.unique()
            print('dates: {}'.format(dates))
            for date in dates[1:]:
                tmp = order_book_data.loc[order_book_data['system_time'].dt.date == date]
                self.export_to_csv(tmp, filename='{}_{}'.format(query['ccy'][0], date), compress=True)

        elapsed = (dt.now(tz=TIMEZONE) - start_time).seconds
        print('***\nSimulator.extract_features() executed in %i seconds\n***' % elapsed)

    def get_orderbook_snapshot_history(self, query):
        """
        Function to replay historical market data and generate
        the features used for reinforcement learning & training.

        NOTE:
        The query can either be a single Coinbase CCY, or both Coinbase and Bitfinex,
        but it cannot be only a Biftinex CCY. Later releases of this repo will
        support Bitfinex only orderbook reconstruction.

        :param query: (dict) query for finding tick history in Arctic TickStore
        :return: (list of arrays) snapshots of limit order books using a stationary feature set
        """
        tick_history = self.get_tick_history(query=query)
        loop_length = tick_history.shape[0]

        coinbase_tick_counter = 0
        snapshot_list = list()
        last_snapshot_time = None

        include_bitfinex = len(query['ccy']) > 1

        coinbase_order_book = CoinbaseOrderBook(query['ccy'][0])
        bitfinex_order_book = BitfinexOrderBook(query['ccy'][1]) if include_bitfinex else None

        start_time = dt.now(TIMEZONE)
        print('Starting get_orderbook_snapshot_history() loop with %i ticks for %s' %
              (loop_length, query['ccy']))

        for idx, tx in enumerate(tick_history.itertuples()):

            tick = tx._asdict()

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
                # multiple = diff // 250000  # 250000 is 250 milliseconds, or 4x a second
                multiple = diff // 500000  # 500000 is 500 milliseconds, or 2x a second

                if multiple >= 1:  # if there is a pause in incoming data, continue to create order book snapshots

                    if include_bitfinex:
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
                                last_snapshot_time += timedelta(milliseconds=500)  # 250)

                            else:
                                last_snapshot_time += timedelta(milliseconds=500)  # 250)
                    else:  # do not include bitfinex
                        for _ in range(multiple):
                            if coinbase_order_book.done_warming_up():
                                coinbase_order_book_snapshot = coinbase_order_book.render_book()
                                snapshot_list.append(list(np.hstack((new_tick_time,  # tick time
                                                                     coinbase_order_book.midpoint,  # midpoint price
                                                                     coinbase_order_book_snapshot))))  # longs/shorts
                                last_snapshot_time += timedelta(milliseconds=500)  # 250)

                            else:
                                last_snapshot_time += timedelta(milliseconds=500)  # 250)

            # incoming tick is from Bitfinex exchange
            elif include_bitfinex & bitfinex_order_book.done_warming_up():
                bitfinex_order_book.new_tick(tick)

            # periodically print number of steps completed
            if idx % 250000 == 0:
                elapsed = (dt.now(TIMEZONE) - start_time).seconds
                print('...completed %i loops in %i seconds' % (idx, elapsed))

            idx += 1

        elapsed = (dt.now(TIMEZONE) - start_time).seconds
        print('Completed run_simulation() with %i ticks in %i seconds at %i ticks/second' %
              (loop_length, elapsed, loop_length//elapsed))

        orderbook_snapshot_history = pd.DataFrame(snapshot_list,
                                                  columns=self.get_feature_labels(
                                                      include_system_time=True,
                                                      include_bitfinex=include_bitfinex))
        orderbook_snapshot_history = orderbook_snapshot_history.dropna(axis=0)

        return orderbook_snapshot_history
