from arctic import Arctic, TICK_STORE
import pytz as tz
from datetime import datetime as dt
from common_components import configs


class Database(object):

    def __init__(self, sym, exchange):
        self.counter = 0
        self.data = list()
        self.sym = sym
        self.tz = tz.utc
        self.exchange = exchange
        if configs.RECORD_DATA:
            self.db = Arctic(configs.MONGO_ENDPOINT)
            self.db.initialize_library(configs.ARCTIC_NAME, lib_type=TICK_STORE)
            self.collection = self.db[configs.ARCTIC_NAME]
            print('%s is recording %s\n' % (self.exchange, self.sym))
        else:
            self.db = None
            self.collection = None

    def new_tick(self, msg):
        if self.db is not None:
            self.counter += 1
            msg['index'] = dt.now(tz=self.tz)
            self.data.append(msg)
            if self.counter % configs.CHUNK_SIZE == 0:
                print('%s added %i msgs to Arctic' % (self.sym, self.counter))
                self.collection.write(self.sym, self.data)
                self.counter = 0
                self.data.clear()
