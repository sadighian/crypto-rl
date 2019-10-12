from datetime import datetime as dt
from datetime import timedelta
from dateutil.parser import parse
import numpy as np
import pandas as pd
import os
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from indicators import apply_ema_all_data, load_ema, reset_ema
from data_recorder.coinbase_connector.coinbase_orderbook import CoinbaseOrderBook
from data_recorder.bitfinex_connector.bitfinex_orderbook import BitfinexOrderBook
from data_recorder.database.database import Database
from configurations.configs import TIMEZONE, MAX_BOOK_ROWS, INCLUDE_ORDERFLOW, \
    SNAPSHOT_RATE_IN_MICROSECONDS


class Simulator(object):

    def __init__(self, z_score=True, alpha=None):
        """
        Simulator constructor
        :param z_score: If TRUE, normalize data with z-score,
                        ELSE use min-max scaler
        """
        self._scaler = StandardScaler() if z_score else MinMaxScaler()
        self.cwd = os.path.dirname(os.path.realpath(__file__))
        self.ema = load_ema(alpha=alpha)
        self.alpha = alpha
        self.db = Database(sym='None', exchange='None', record_data=False)

    def __str__(self):
        return 'Simulator: [ scaler={} | ema={} ]'.format(
            self._scaler.__class__, self.ema)

    @staticmethod
    def get_feature_labels(include_system_time: bool = True,
                           include_bitfinex: bool = True,
                           include_order_flow: bool = INCLUDE_ORDERFLOW,
                           include_imbalances: bool = True,
                           include_spread: bool = False,
                           include_ema=None):
        """
        Function to create the features' labels
        :param include_bitfinex: (boolean) If TRUE, Bitfinex's LOB data
                is included in the dataset, in addition to Coinbase-Pro
        :param include_system_time: True/False
                (False removes the system_time column)
        :param include_order_flow: True/False
                if TRUE, order arrival metrics are included in the feature set
        :param include_imbalances: True/False
                if TRUE, order volume imbalances at level are included in the feature set
        :param include_spread: True/False
                if TRUE, order spread column is included
        :param include_ema: None, float, or list
                if list, then append alphas to each column
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
            for feature in ['notional', 'distance']:
                for side in ['bid', 'ask']:
                    if side == 'bid':
                        for level in reversed(range(MAX_BOOK_ROWS)):
                            columns.append(('%s_%s_%s_%i' %
                                            (exchange, side, feature, level)))
                    else:
                        for level in range(MAX_BOOK_ROWS):
                            columns.append(('%s_%s_%s_%i' %
                                            (exchange, side, feature, level)))

            for trade_side in ['buys', 'sells']:
                columns.append('%s_%s' % (exchange, trade_side))

            if include_order_flow:
                for feature in ['cancel_notional', 'limit_notional', 'market_notional']:
                    for side in ['bid', 'ask']:
                        if side == 'bid':
                            for level in reversed(range(MAX_BOOK_ROWS)):
                                columns.append(('%s_%s_%s_%i' %
                                                (exchange, side, feature, level)))
                        else:
                            for level in range(MAX_BOOK_ROWS):
                                columns.append(('%s_%s_%s_%i' %
                                                (exchange, side, feature, level)))

            if include_spread:
                columns.append('{}_spread'.format(exchange))

            if include_imbalances:
                for level in range(MAX_BOOK_ROWS):
                    columns.append('notional_imbalance_{}'.format(level))
                columns.append('notional_imbalance_mean')
                columns.append('notional_imbalance_std')

        if isinstance(include_ema, list):
            tmp = list()
            for ema in include_ema:
                for col in columns:
                    if col == 'system_time':
                        continue
                    tmp.append('{}_{}'.format(col, ema))
            if include_system_time:
                tmp.insert(0, 'system_time')
            columns = tmp

        return columns

    def export_to_csv(self, data: pd.DataFrame, filename='BTC-USD_2019-01-01',
                      compress=True):
        """
        Export data within a Panda dataframe to a csv
        :param data: (panda.DataFrame) historical tick data
        :param filename: CCY_YYYY-MM-DD
        :param compress: Default True. If True, compress with xz
        :return: void
        """
        start_time = dt.now(tz=TIMEZONE)

        sub_folder = os.path.join(self.cwd, 'data_exports', filename) + '.csv'

        if compress:
            sub_folder += '.xz'
            data.to_csv(path_or_buf=sub_folder, index=False, compression='xz')
        else:
            data.to_csv(path_or_buf=sub_folder, index=False)

        elapsed = (dt.now(tz=TIMEZONE) - start_time).seconds
        print('Exported %s with %i rows in %i seconds' %
              (sub_folder, data.shape[0], elapsed))

    @staticmethod
    def import_csv(filename: str) -> pd.DataFrame:
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

    def fit_scaler(self, orderbook_snapshot_history: pd.DataFrame):
        """
        Scale limit order book data for the neural network
        :param orderbook_snapshot_history: Limit order book data
                from the previous day
        :return: (void)
        """
        self._scaler.fit(orderbook_snapshot_history)

    def scale_data(self, data: pd.DataFrame):
        """
        Normalize data
        :param data: (np.array) all data in environment
        :return: (np.array) normalized observation space
        """
        return self._scaler.transform(data)

    @staticmethod
    def _midpoint_diff(data: pd.DataFrame):
        """
        Take log difference of midpoint prices
                log(price t) - log(price t-1)
        :param data: (pd.DataFrame) raw data from LOB snapshots
        :return: (pd.DataFrame) with midpoint prices normalized
        """
        data['coinbase_midpoint'] = np.log(data['coinbase_midpoint'].values)
        data['coinbase_midpoint'] = (
                data['coinbase_midpoint'] - data['coinbase_midpoint'].shift(1)).fillna(
            method='bfill')
        return data

    @staticmethod
    def _spread_calc(data: pd.DataFrame) -> pd.DataFrame:
        """
        Derive the spread and normalize it by a multiple of the market order fee.
        :param data: (pd.DataFrame) data set containing a bid and ask
        :return: data with spread added as the last column
        """
        # calculate the spread in real terms ('+' because bid_distances are all negative)
        data['coinbase_spread'] = data['coinbase_ask_distance_0'].values + \
                                  data['coinbase_bid_distance_0'].values
        return data

    @staticmethod
    def _get_order_imbalance(data: pd.DataFrame):
        """
        Calculate order imbalances per price level, their mean & standard deviation.

        Order Imbalances are calculated by:
            = (bid_quantity - ask_quantity) / (bid_quantity + ask_quantity)

        ...thus scale from [-1, 1].

        :param data: raw/unnormalized LOB snapshot data
        :return: (pd.DataFrame) order imbalances at N-levels, the mean & std imbalance
        """
        # create the column names for making a data frame (also used for debugging)
        bid_notional_columns, ask_notional_columns, imbalance_columns = [], [], []
        for i in range(MAX_BOOK_ROWS):
            bid_notional_columns.append('coinbase_bid_notional_{}'.format(i))
            ask_notional_columns.append('coinbase_ask_notional_{}'.format(i))
            imbalance_columns.append('notional_imbalance_{}'.format(i))
        # acquire bid and ask notional data
        bid_notional = data[bid_notional_columns].values[::-1]  # reverse the bids to
        # ascending order, so that they align with the asks
        ask_notional = data[ask_notional_columns].values
        # calculate the order imbalance
        imbalances = (bid_notional - ask_notional) / (bid_notional + ask_notional)
        imbalances = pd.DataFrame(imbalances, columns=imbalance_columns).fillna(0.)
        # add meta data to features (mean and std)
        imbalances['notional_imbalance_mean'] = imbalances[imbalance_columns].mean(axis=1)
        imbalances['notional_imbalance_std'] = imbalances[imbalance_columns].std(axis=1)
        return imbalances

    def load_environment_data(self, fitting_file: str, testing_file: str,
                              include_imbalances: bool = True, as_pandas: bool = False):
        """
        Import and scale environment data set with prior day's data.

        Midpoint gets log-normalized:
            log(price t) - log(price t-1)

        :param fitting_file: prior trading day
        :param testing_file: current trading day
        :param include_imbalances: if TRUE, include LOB imbalances
        :param as_pandas: if TRUE, return data as DataFrame, otherwise np.array
        :return: (pd.DataFrame or np.array) scaled environment data
        """
        # import data used to fit scaler
        fitting_data_filepath = os.path.join(self.cwd, 'data_exports', fitting_file)
        fitting_data = self.import_csv(filename=fitting_data_filepath)
        # check if bitfinex data is in the data set
        include_bitfinex = 'bitfinex' in fitting_data.columns.tolist()
        # carry on with data import process
        fitting_data = self._midpoint_diff(data=fitting_data)  # normalize midpoint
        fitting_data = self._spread_calc(data=fitting_data)  # normalize spread
        fitting_data = apply_ema_all_data(ema=self.ema, data=fitting_data)
        self.fit_scaler(fitting_data)
        del fitting_data

        # import data to normalize and use in environment
        data_used_in_environment = os.path.join(self.cwd, 'data_exports', testing_file)
        data = self.import_csv(filename=data_used_in_environment)
        midpoint_prices = data['coinbase_midpoint']

        normalized_data = self._midpoint_diff(data.copy(deep=True))
        normalized_data = self._spread_calc(data=normalized_data)  # normalize spread
        normalized_data = apply_ema_all_data(ema=self.ema, data=normalized_data)

        normalized_data = self.scale_data(normalized_data)
        normalized_data = np.clip(normalized_data, -10, 10)
        normalized_data = pd.DataFrame(normalized_data, columns=self.get_feature_labels(
            include_system_time=False, include_bitfinex=include_bitfinex,
            include_spread=True,
            include_imbalances=False, include_ema=self.alpha))

        if include_imbalances:
            print('Adding order imbalances...')
            # Note: since order imbalance data is scaled [-1, 1], we do not apply
            # z-score to the imbalance data
            imbalance_data = self._get_order_imbalance(data=data)
            imbalance_data = apply_ema_all_data(ema=reset_ema(self.ema),
                                                data=imbalance_data)
            normalized_data = pd.concat((normalized_data, imbalance_data), axis=1)

        if as_pandas is False:
            midpoint_prices = midpoint_prices.values
            data = data.values
            normalized_data = normalized_data.values

        return midpoint_prices, data, normalized_data

    @staticmethod
    def _get_microsecond_delta(new_tick_time: dt, last_snapshot_time: dt):
        """
        Calculate difference between two consecutive ticks.
        Note: only tracks timedelta for up to a minute.
        :param new_tick_time: datetime of incoming tick
        :param last_snapshot_time: datetime of last LOB snapshot
        :return: (int) delta between ticks
        """
        if last_snapshot_time > new_tick_time:
            return -1
        snapshot_tick_time_delta = new_tick_time - last_snapshot_time
        seconds = snapshot_tick_time_delta.seconds * 1000000
        microseconds = snapshot_tick_time_delta.microseconds
        # print("seconds={} | microseconds={}".format(seconds, microseconds))
        return seconds + microseconds

    def get_orderbook_snapshot_history(self, query: dict):
        """
        Function to replay historical market data and generate
        the features used for reinforcement learning & training.

        NOTE:
        The query can either be a single Coinbase CCY, or both Coinbase and Bitfinex,
        but it cannot be only a Biftinex CCY. Later releases of this repo will
        support Bitfinex only orderbook reconstruction.

        :param query: (dict) query for finding tick history in Arctic TickStore
        :return: (pd.DataFrame) snapshots of limit order books using a
                stationary feature set
        """
        self.db.init_db_connection()
        tick_history = self.db.get_tick_history(query=query)
        if tick_history is None:
            print("Query returned no data: {}".format(query))
            return None

        loop_length = tick_history.shape[0]

        # number of microseconds between LOB snapshots
        snapshot_interval_milliseconds = SNAPSHOT_RATE_IN_MICROSECONDS // 1000

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

        # loop through all ticks returned from the Arctic Tick Store query.
        for count, tx in enumerate(tick_history.itertuples()):

            # periodically print number of steps completed
            if count % 250000 == 0:
                elapsed = (dt.now(TIMEZONE) - start_time).seconds
                print('...completed %i loops in %i seconds' % (count, elapsed))

            # convert to dictionary for processing
            tick = tx._asdict()

            # determine if incoming tick is from coinbase or bitfinex
            coinbase = True if tick['product_id'] == coinbase_order_book.sym else \
                False

            # filter out bad ticks
            if 'type' not in tick:
                continue

            # flags for a order book reset
            if tick['type'] in ['load_book', 'book_loaded', 'preload']:
                if coinbase:
                    coinbase_order_book.new_tick(tick)
                else:
                    bitfinex_order_book.new_tick(tick)
                # skip to next loop
                continue

            # incoming tick is for coinbase LOB
            if coinbase:
                # check if the LOB is pre-loaded, if not skip message and do NOT process.
                if coinbase_order_book.done_warming_up() is False:
                    print("coinbase_order_book not done warming up: {}".format(tick))
                    continue

                # timestamp for incoming tick
                new_tick_time = parse(tick.get('time'))

                # remove ticks without timestamps (should not exist/happen)
                if new_tick_time is None:
                    print('No tick time: {}'.format(tick))
                    continue

                # initialize the LOB snapshot timer
                if last_snapshot_time is None:
                    # process first ticks and check if they're stale ticks; if so,
                    # skip to the next loop.
                    coinbase_order_book.new_tick(tick)
                    last_coinbase_tick_time = coinbase_order_book.last_tick_time
                    if last_coinbase_tick_time is None:
                        continue
                    last_coinbase_tick_time_dt = parse(last_coinbase_tick_time)
                    last_snapshot_time = last_coinbase_tick_time_dt
                    print('{} first tick: {} | Sequence: {}'.format(
                        coinbase_order_book.sym, new_tick_time,
                        coinbase_order_book.sequence))
                    # skip to next loop
                    continue

                # calculate the amount of time between the incoming
                #   tick and tick received before that
                diff = self._get_microsecond_delta(new_tick_time, last_snapshot_time)

                # update the LOB, but do not take a LOB snapshot if the tick time is
                # out of sequence. This occurs when pre-loading a LOB with stale tick
                # times in general.
                if diff == -1:
                    coinbase_order_book.new_tick(tick)
                    continue

                # derive the number of LOB snapshot insertions for the data buffer.
                multiple = diff // SNAPSHOT_RATE_IN_MICROSECONDS  # 1000000 is 1 second

                # proceed if we have one or more insertions to make
                if multiple <= 0:
                    coinbase_order_book.new_tick(tick)
                    continue

                # check to include Bitfinex data in features.
                if include_bitfinex:
                    # if bitfinex's LOB is still loading, do NOT export snapshots
                    # of coinbase in the meantime and continue to next loop.
                    if bitfinex_order_book.done_warming_up() is False:
                        print("bitfinex_order_book not done warming up: {}".format(
                            tick))
                        coinbase_order_book.new_tick(tick)
                        # update the LOB snapshot tracker.
                        for _ in range(multiple):
                            last_snapshot_time += timedelta(
                                milliseconds=snapshot_interval_milliseconds)
                        # move to next loop and see if bitfinex's LOB is ready then.
                        continue

                    # since both coinbase and bitfinex LOBs are assumed to be
                    # pre-loaded at this point, we can proceed to export snapshots
                    # of the LOB, even if there has been a 'long' duration between
                    # consecutive ticks.
                    coinbase_order_book_snapshot = coinbase_order_book.render_book()
                    bitfinex_order_book_snapshot = bitfinex_order_book.render_book()
                    midpoint_delta = coinbase_order_book.midpoint - \
                        bitfinex_order_book.midpoint

                    # update the LOB snapshot time-delta AND add LOB snapshots to the
                    # data buffer.
                    for i in range(multiple):
                        last_snapshot_time += timedelta(
                            milliseconds=snapshot_interval_milliseconds)
                        snapshot_list.append(np.hstack((
                            last_snapshot_time,
                            coinbase_order_book.midpoint,  # midpoint price
                            midpoint_delta,  # price delta between exchanges
                            coinbase_order_book_snapshot,
                            bitfinex_order_book_snapshot)))  # longs/shorts

                    # update order book with most recent tick now, so the snapshots
                    # are up to date for the next iteration of the loop.
                    coinbase_order_book.new_tick(tick)
                    continue
                else:  # do not include bitfinex
                    coinbase_order_book_snapshot = coinbase_order_book.render_book()
                    for i in range(multiple):
                        last_snapshot_time += timedelta(
                            milliseconds=snapshot_interval_milliseconds)
                        snapshot_list.append(np.hstack((
                            last_snapshot_time,
                            coinbase_order_book.midpoint,
                            coinbase_order_book_snapshot)))

                    # update order book with most recent tick now, so the snapshots
                    # are up to date for the next iteration of the loop.
                    coinbase_order_book.new_tick(tick)
                    continue

            # incoming tick is from Bitfinex exchange
            elif include_bitfinex and bitfinex_order_book.done_warming_up():
                bitfinex_order_book.new_tick(tick)
                continue

        elapsed = (dt.now(TIMEZONE) - start_time).seconds
        print('Completed run_simulation() with %i ticks in %i seconds '
              'at %i ticks/second'
              % (loop_length, elapsed, loop_length//elapsed))

        orderbook_snapshot_history = pd.DataFrame(
            snapshot_list,
            columns=self.get_feature_labels(
                include_system_time=True, include_spread=False,
                include_bitfinex=include_bitfinex, include_order_flow=INCLUDE_ORDERFLOW,
                include_imbalances=False, include_ema=self.alpha))
        orderbook_snapshot_history = orderbook_snapshot_history.dropna(axis=0)

        return orderbook_snapshot_history

    def extract_features(self, query: dict):
        """
        Create and export limit order book data to csv. This function
        exports multiple days of data and ensures each day starts and
        ends exactly on time.
        :param query: (dict) ccy=sym, daterange=(YYYYMMDD,YYYYMMDD)
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
                                   compress=True)

        elapsed = (dt.now(tz=TIMEZONE) - start_time).seconds
        print('***\nSimulator.extract_features() executed in %i seconds\n***'
              % elapsed)
