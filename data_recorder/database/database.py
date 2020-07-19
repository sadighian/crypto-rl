from datetime import datetime as dt
from typing import Union

import numpy as np
import pandas as pd
from arctic import Arctic, TICK_STORE
from arctic.date import DateRange
from pymongo.errors import PyMongoError

from configurations import (
    ARCTIC_NAME, BATCH_SIZE, LOGGER, MONGO_ENDPOINT, RECORD_DATA, TIMEZONE,
)


class Database(object):

    def __init__(self, sym: str, exchange: str, record_data: bool = RECORD_DATA):
        """
        Database constructor.
        """
        self.counter = 0
        self.data = list()
        self.tz = TIMEZONE
        self.sym = sym
        self.exchange = exchange
        self.recording = record_data
        self.db = self.collection = None
        if self.recording:
            LOGGER.info('\nDatabase: [%s is recording %s]\n' % (self.exchange, self.sym))

    def init_db_connection(self) -> None:
        """
        Initiate database connection to Arctic.

        :return: (void)
        """
        LOGGER.info("init_db_connection for {}...".format(self.sym))
        try:
            self.db = Arctic(MONGO_ENDPOINT)
            self.db.initialize_library(ARCTIC_NAME, lib_type=TICK_STORE)
            self.collection = self.db[ARCTIC_NAME]
        except PyMongoError as e:
            LOGGER.warn("Database.PyMongoError() --> {}".format(e))

    def new_tick(self, msg: dict) -> None:
        """
        If RECORD_DATA is TRUE, add streaming ticks to a list
        After the list has accumulated BATCH_SIZE ticks, insert batch into
        the Arctic Tick Store.

        :param msg: incoming tick
        :return: void
        """

        if self.recording is False:
            return

        self.counter += 1
        msg['index'] = dt.now(tz=self.tz)
        msg['system_time'] = str(msg['index'])
        self.data.append(msg)
        if self.counter % BATCH_SIZE == 0:
            self.collection.write(self.sym, self.data)
            LOGGER.info('{} added {} msgs to Arctic'.format(self.sym, self.counter))
            self.counter = 0
            self.data.clear()

    def _query_arctic(self,
                      ccy: str,
                      start_date: int,
                      end_date: int) -> Union[pd.DataFrame, None]:
        """
        Query database and return LOB messages starting from LOB reconstruction.

        :param ccy: currency symbol
        :param start_date: YYYYMMDD start date
        :param end_date: YYYYMMDD end date
        :return: (pd.DataFrame) results found in database
        """
        assert self.collection is not None, \
            "Arctic.Collection() must not be null."

        start_time = dt.now(tz=self.tz)

        try:
            LOGGER.info('\nGetting {} data from Arctic Tick Store...'.format(ccy))
            cursor = self.collection.read(symbol=ccy,
                                          date_range=DateRange(start_date, end_date))

            # filter ticks for the first LOAD_BOOK message
            #   (starting point for order book reconstruction)
            # min_datetime = cursor.loc[cursor.type == 'load_book'].index[0]
            dates = np.unique(cursor.loc[cursor.type == 'load_book'].index.date)
            start_index = cursor.loc[((cursor.index.date == dates[0]) &
                                      (cursor.type == 'load_book'))].index[-1]
            # cursor = cursor.loc[cursor.index >= min_datetime]
            cursor = cursor.loc[cursor.index >= start_index]

            elapsed = (dt.now(tz=self.tz) - start_time).seconds
            LOGGER.info('Completed querying %i %s records in %i seconds' %
                        (cursor.shape[0], ccy, elapsed))

        except Exception as ex:
            cursor = None
            LOGGER.warn('Simulator._query_arctic() thew an exception: \n%s' % str(ex))

        return cursor

    def get_tick_history(self, query: dict) -> Union[pd.DataFrame, None]:
        """
        Function to query the Arctic Tick Store and...
        1.  Return the specified historical data for a given set of securities
            over a specified amount of time
        2.  Convert the data returned from the query from a panda to a list of dicts
            and while doing so, allocate the work across all available CPU cores

        :param query: (dict) of the query parameters
            - ccy: list of symbols
            - startDate: int YYYYMMDD start date
            - endDate: int YYYYMMDD end date
        :return: list of dicts, where each dict is a tick that was recorded
        """
        start_time = dt.now(tz=self.tz)

        assert self.recording is False, "RECORD_DATA must be set to FALSE to replay data"
        cursor = self._query_arctic(**query)
        if cursor is None:
            LOGGER.info('\nNothing returned from Arctic for the query: %s\n...Exiting...'
                        % str(query))
            return

        elapsed = (dt.now(tz=self.tz) - start_time).seconds
        LOGGER.info('***Completed get_tick_history() in %i seconds***' % elapsed)

        return cursor
