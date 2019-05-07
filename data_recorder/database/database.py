from configurations.configs import TIMEZONE, RECORD_DATA, MONGO_ENDPOINT, ARCTIC_NAME, BATCH_SIZE
from arctic import Arctic, TICK_STORE
from datetime import datetime as dt


class Database(object):

    def __init__(self, sym, exchange):
        self.counter = 0
        self.data = list()
        self.sym = sym
        self.tz = TIMEZONE
        self.exchange = exchange
        if RECORD_DATA:
            self.db = Arctic(MONGO_ENDPOINT)
            self.db.initialize_library(ARCTIC_NAME, lib_type=TICK_STORE)
            self.collection = self.db[ARCTIC_NAME]
            print('\n%s is recording %s\n' % (self.exchange, self.sym))
        else:
            self.db = None
            self.collection = None

    def new_tick(self, msg):
        """
        If RECORD_DATA is TRUE, add streaming ticks to a list
        After the list has accumulated BATCH_SIZE ticks, insert batch into
        the Arctic Tick Store
        :param msg: incoming tick
        :return: void
        """
        if self.db is not None:
            self.counter += 1
            msg['index'] = dt.now(tz=self.tz)
            msg['system_time'] = str(msg['index'])
            self.data.append(msg)
            if self.counter % BATCH_SIZE == 0:
                print('%s added %i msgs to Arctic' % (self.sym, self.counter))
                self.collection.write(self.sym, self.data)
                self.counter = 0
                self.data.clear()
