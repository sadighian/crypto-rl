from abc import ABC, abstractmethod
from datetime import datetime as dt

from pymongo import MongoClient


class ABook(ABC):

    def __init__(self, sym):
        self.sym = sym
        self.timer_frequency = 0.2
        self.recording = False
        self.last_time = dt.now()
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
    def best_ask(selfs):
        pass

    @abstractmethod
    def new_tick(self, msg):
        pass

    @abstractmethod
    def timer_worker(self):
        pass
