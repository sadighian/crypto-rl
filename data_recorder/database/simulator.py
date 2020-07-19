import os
from datetime import datetime as dt
from datetime import timedelta
from typing import Type, Union

import numpy as np
import pandas as pd
from dateutil.parser import parse

from configurations import DATA_PATH, LOGGER, SNAPSHOT_RATE_IN_MICROSECONDS, TIMEZONE
from data_recorder.bitfinex_connector.bitfinex_orderbook import BitfinexOrderBook
from data_recorder.coinbase_connector.coinbase_orderbook import CoinbaseOrderBook
from data_recorder.database.database import Database

DATA_EXPORTS_PATH = DATA_PATH


def _get_exchange_from_symbol(symbol: str) -> str:
    """
    Get exchange name given an instrument name.

    :param symbol: instrument name or currency pair
    :return: exchange name
    """
    symbol_inventory = dict({
        'BTC-USD': 'coinbase',
        'ETH-USD': 'coinbase',
        'LTC-USD': 'coinbase',
        'tBTCUSD': 'bitfinex',
        'tETHUSD': 'bitfinex',
        'tLTCUSD': 'bitfinex',
        'XBTUSD': 'bitmex',
        'XETUSD': 'bitmex',
    })
    return symbol_inventory[symbol]


def _get_orderbook_from_exchange(exchange: str) -> \
        Type[Union[CoinbaseOrderBook, BitfinexOrderBook]]:
    """
    Get order book given an exchange name.

    :param exchange: name of exchange ['bitfinex' or 'coinbase']
    :return: order book for 'exchange'
    """
    return dict(coinbase=CoinbaseOrderBook, bitfinex=BitfinexOrderBook)[exchange]


def get_orderbook_from_symbol(symbol: str) -> \
        Type[Union[CoinbaseOrderBook, BitfinexOrderBook]]:
    """
    Get order book given an instrument name.

    :param symbol: instrument name
    :return: order book for 'symbol'
    """
    return _get_orderbook_from_exchange(exchange=_get_exchange_from_symbol(symbol=symbol))


class Simulator(object):

    def __init__(self):
        """
        Simulator constructor.
        """
        self.cwd = os.path.dirname(os.path.realpath(__file__))
        self.db = Database(sym='None', exchange='None', record_data=False)

    def __str__(self):
        return 'Simulator: [ db={} ]'.format(self.db)

    @staticmethod
    def export_to_csv(data: pd.DataFrame,
                      filename: str = 'BTC-USD_2019-01-01',
                      compress: bool = True) -> None:
        """
        Export data within a Panda DataFrame to a csv.

        :param data: (panda.DataFrame) historical tick data
        :param filename: CCY_YYYY-MM-DD
        :param compress: Default True. If True, compress with xz
        """
        start_time = dt.now(tz=TIMEZONE)

        sub_folder = os.path.join(DATA_PATH, filename) + '.csv'

        if compress:
            sub_folder += '.xz'
            data.to_csv(path_or_buf=sub_folder, index=False, compression='xz')
        else:
            data.to_csv(path_or_buf=sub_folder, index=False)

        elapsed = (dt.now(tz=TIMEZONE) - start_time).seconds
        LOGGER.info('Exported %s with %i rows in %i seconds' %
                    (sub_folder, data.shape[0], elapsed))

    @staticmethod
    def get_ema_labels(features_list: list, ema_list: list, include_system_time: bool):
        """
        Get a list of column labels for EMA values in a list.
        """
        assert isinstance(ema_list, list) is True, \
            "Error: EMA_LIST must be a list data type, not {}".format(type(ema_list))

        ema_labels = list()

        for ema in ema_list:
            for col in features_list:
                if col == 'system_time':
                    continue
                ema_labels.append('{}_{}'.format(col, ema))

        if include_system_time:
            ema_labels.insert(0, 'system_time')

        return ema_labels

    @staticmethod
    def _get_microsecond_delta(new_tick_time: dt, last_snapshot_time: dt) -> int:
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

        return seconds + microseconds

    def get_orderbook_snapshot_history(self, query: dict) -> pd.DataFrame or None:
        """
        Function to replay historical market data and generate the features used for
        reinforcement learning & training.

        NOTE:
            The query can either be a single Coinbase CCY, or both Coinbase and Bitfinex,
            but it cannot be only a Bitfinex CCY. Later releases of this repo will
            support Bitfinex only order book reconstruction.

        :param query: (dict) query for finding tick history in Arctic TickStore
        :return: (pd.DataFrame) snapshots of limit order books using a
                stationary feature set
        """
        self.db.init_db_connection()

        tick_history = self.db.get_tick_history(query=query)
        if tick_history is None:
            LOGGER.warn("Query returned no data: {}".format(query))
            return None

        loop_length = tick_history.shape[0]

        # number of microseconds between LOB snapshots
        snapshot_interval_milliseconds = SNAPSHOT_RATE_IN_MICROSECONDS // 1000

        snapshot_list = list()
        last_snapshot_time = None
        tick_types_for_warm_up = {'load_book', 'book_loaded', 'preload'}

        instrument_name = query['ccy'][0]
        assert isinstance(instrument_name, str), \
            "Error: instrument_name must be a string, not -> {}".format(
                type(instrument_name))

        LOGGER.info('querying {}'.format(instrument_name))

        order_book = get_orderbook_from_symbol(symbol=instrument_name)(
            sym=instrument_name)

        start_time = dt.now(tz=TIMEZONE)
        LOGGER.info('Starting get_orderbook_snapshot_history() loop with %i ticks for %s'
                    % (loop_length, query['ccy']))

        # loop through all ticks returned from the Arctic Tick Store query.
        for count, tx in enumerate(tick_history.itertuples()):

            # periodically print number of steps completed
            if count % 250000 == 0:
                elapsed = (dt.now(tz=TIMEZONE) - start_time).seconds
                LOGGER.info('...completed %i loops in %i seconds' % (count, elapsed))

            # convert to dictionary for processing
            tick = tx._asdict()

            # filter out bad ticks
            if 'type' not in tick:
                continue

            # flags for a order book reset
            if tick['type'] in tick_types_for_warm_up:
                order_book.new_tick(msg=tick)
                continue

            # check if the LOB is pre-loaded, if not skip message and do NOT process.
            if order_book.done_warming_up is False:
                LOGGER.info(
                    "{} order book is not done warming up: {}".format(
                        instrument_name, tick))
                continue

            # timestamp for incoming tick
            new_tick_time = parse(tick.get('system_time'))

            # remove ticks without timestamps (should not exist/happen)
            if new_tick_time is None:
                LOGGER.info('No tick time: {}'.format(tick))
                continue

            # initialize the LOB snapshot timer
            if last_snapshot_time is None:
                # process first ticks and check if they're stale ticks; if so,
                # skip to the next loop.
                order_book.new_tick(tick)

                last_tick_time = order_book.last_tick_time
                if last_tick_time is None:
                    continue

                last_tick_time_dt = parse(last_tick_time)
                last_snapshot_time = last_tick_time_dt
                LOGGER.info('{} first tick: {} '.format(order_book.sym, new_tick_time))
                # skip to next loop
                continue

            # calculate the amount of time between the incoming
            #   tick and tick received before that
            diff = self._get_microsecond_delta(new_tick_time, last_snapshot_time)

            # update the LOB, but do not take a LOB snapshot if the tick time is
            # out of sequence. This occurs when pre-loading a LOB with stale tick
            # times in general.
            if diff == -1:
                order_book.new_tick(msg=tick)
                continue

            # derive the number of LOB snapshot insertions for the data buffer.
            multiple = diff // SNAPSHOT_RATE_IN_MICROSECONDS  # 1000000 is 1 second

            # proceed if we have one or more insertions to make
            if multiple <= 0:
                order_book.new_tick(msg=tick)
                continue

            order_book_snapshot = order_book.render_book()
            for i in range(multiple):
                last_snapshot_time += timedelta(
                    milliseconds=snapshot_interval_milliseconds)
                snapshot_list.append(np.hstack((last_snapshot_time, order_book_snapshot)))

            # update order book with most recent tick now, so the snapshots
            # are up to date for the next iteration of the loop.
            order_book.new_tick(msg=tick)
            continue

        elapsed = max((dt.now(tz=TIMEZONE) - start_time).seconds, 1)
        LOGGER.info('Completed run_simulation() with %i ticks in %i seconds '
                    'at %i ticks/second'
                    % (loop_length, elapsed, loop_length // elapsed))

        orderbook_snapshot_history = pd.DataFrame(
            data=snapshot_list,
            columns=['system_time'] + order_book.render_lob_feature_names()
        )

        # remove NAs from data set (and print the amount)
        before_shape = orderbook_snapshot_history.shape[0]
        orderbook_snapshot_history = orderbook_snapshot_history.dropna(axis=0)
        difference_in_records = orderbook_snapshot_history.shape[0] - before_shape
        LOGGER.info("{} {} rows due to NA values".format(
            'Dropping' if difference_in_records <= 0 else 'Adding',
            abs(difference_in_records))
        )

        return orderbook_snapshot_history

    def extract_features(self, query: dict) -> None:
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
            LOGGER.info('dates: {}'.format(dates))
            for date in dates[:]:
                # for date in dates[1:]:
                tmp = order_book_data.loc[order_book_data['system_time'].dt.date == date]
                self.export_to_csv(
                    tmp, filename='{}_{}'.format(query['ccy'][0], date), compress=True)

        elapsed = (dt.now(tz=TIMEZONE) - start_time).seconds
        LOGGER.info('***\nSimulator.extract_features() executed in %i seconds\n***'
                    % elapsed)
