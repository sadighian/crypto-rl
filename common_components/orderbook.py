from abc import ABC, abstractmethod
from common_components.database import Database


class OrderBook(ABC):

    def __init__(self, ccy, exchange):
        self.sym = ccy
        self.exchange = exchange
        self.db = Database(ccy, exchange)

    def __str__(self):
        return '%s  |  %s' % (self.sym, self.exchange)

    @abstractmethod
    def new_tick(self, msg):
        pass
