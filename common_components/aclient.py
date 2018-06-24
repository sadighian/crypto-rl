from abc import ABC, abstractmethod
import json
import os
import time
from datetime import datetime as dt
from multiprocessing import JoinableQueue as Queue
from threading import Thread
import websockets


class AClient(ABC):

    def __init__(self, ccy, book):
        self.sym = ccy
        self.book = book(ccy)
        self.queue = Queue(maxsize=1000)
        self.ws = None
        self.retry_counter = 0
        self.last_subscribe_time = None

