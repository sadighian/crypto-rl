from datetime import datetime as dt
import numpy as np
from pymongo.errors import PyMongoError
from arctic import Arctic, TICK_STORE
from arctic.date import DateRange
from configurations.configs import TIMEZONE, RECORD_DATA, MONGO_ENDPOINT, \
    ARCTIC_NAME, BATCH_SIZE


class Database(object):

    def __init__(self, sym: str, exchange: str, record_data: bool = RECORD_DATA):
        self.counter = 0
        self.data = list()
        self.tz = TIMEZONE
        self.sym = sym
        self.exchange = exchange
        self.recording = record_data
        self.db = self.collection = None
        if self.recording:
            print('\nDatabase: [%s is recording %s]\n' % (self.exchange, self.sym))

    def init_db_connection(self):
        """
        Initiate database connection
        :return: (void)
        """
        print("init_db_connection for {}...".format(self.sym))
        try:
            self.db = Arctic(MONGO_ENDPOINT)
            self.db.initialize_library(ARCTIC_NAME, lib_type=TICK_STORE)
            self.collection = self.db[ARCTIC_NAME]
        except PyMongoError as e:
            print("Database.PyMongoError() --> {}".format(e))

    def new_tick(self, msg: dict):
        """
        If RECORD_DATA is TRUE, add streaming ticks to a list
        After the list has accumulated BATCH_SIZE ticks, insert batch into
        the Arctic Tick Store
        :param msg: incoming tick
        :return: void
        """
        if self.recording:
            self.counter += 1
            msg['index'] = dt.now(tz=self.tz)
            msg['system_time'] = str(msg['index'])
            self.data.append(msg)
            if self.counter % BATCH_SIZE == 0:
                print('%s added %i msgs to Arctic' % (self.sym, self.counter))
                self.collection.write(self.sym, self.data)
                self.counter = 0
                self.data.clear()

    def _query_arctic(self, ccy: str, start_date: int, end_date: int):
        """
        Query database and return LOB messages starting from LOB reconstruction
        :param ccy: currency symbol
        :param start_date: YYYYMMDD
        :param end_date: YYYYMMDD
        :return: (pd.DataFrame)
        """
        start_time = dt.now(tz=TIMEZONE)

        if self.collection is None:
            print('exiting from Simulator... no database to query')
            return None

        try:
            print('\nGetting {} tick data from Arctic Tick Store...'.format(ccy))
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

            elapsed = (dt.now(tz=TIMEZONE) - start_time).seconds
            print('Completed querying %i %s records in %i seconds' %
                  (cursor.shape[0], ccy, elapsed))

        except Exception as ex:
            cursor = None
            print('Simulator._query_arctic() thew an exception: \n%s' % str(ex))

        return cursor

    def get_tick_history(self, query: dict):
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

        assert self.recording is False, "RECORD_DATA must be set to FALSE to replay data"
        cursor = self._query_arctic(**query)
        if cursor is None:
            print('\nNothing returned from Arctic for the query: %s\n...Exiting...'
                  % str(query))
            return

        elapsed = (dt.now(tz=TIMEZONE) - start_time).seconds
        print('***\nCompleted get_tick_history() in %i seconds\n***' % elapsed)

        return cursor
