from abc import ABC, abstractmethod
from datetime import datetime as dt
from pymongo import MongoClient


class ABook(ABC):

    def __init__(self, sym):
        self.sym = sym
        self.timer_frequency = 2.0  # 0.2 = 5x second
        self.last_time = dt.now()
        self.recording = False
        self.db = MongoClient('mongodb://localhost:27017')[self.sym] if self.recording else None
        self.trades = dict({
            'upticks': {
                'size': float(0),
                'count': int(0)
            },
            'downticks': {
                'size': float(0),
                'count': int(0)
            }
        })

    def _reset_trades_tracker(self):
        self.trades = dict({
            'upticks': {
                'size': float(0),
                'count': int(0)
            },
            'downticks': {
                'size': float(0),
                'count': int(0)
            }
        })

    @property
    def _get_trades_tracker(self):
        return self.trades

    def record(self, current_time):
        """
        Insert snapshot of limit order book into Mongo DB
        :param current_time: dt.now()
        :return: void
        """
        if self.db is not None:
            current_date = current_time.strftime("%Y-%m-%d")
            self.db[current_date].insert_one(self.render_book())
            self._reset_trades_tracker()
        else:
            pass
            # print(self)

    @abstractmethod
    def clear_book(self):
        pass

    @abstractmethod
    def render_book(self):
        pass

    @abstractmethod
    def best_bid(self):
        pass

    @abstractmethod
    def best_ask(self):
        pass

    @abstractmethod
    def new_tick(self, msg):
        pass

    @abstractmethod
    def timer_worker(self):
        pass
