from abc import ABC, abstractmethod
from datetime import datetime as dt
from sortedcontainers import SortedDict
from common_components import configs
import pandas as pd
import pytz as tz


class Book(ABC):

    def __init__(self, sym, side):
        self.price_dict = SortedDict()
        self.order_map = dict()
        self.side = side
        self.sym = sym
        self.warming_up = True  # this value needs to be set within the orderbook class
        self.max_book_size = configs.MAX_BOOK_ROWS

    def __str__(self):
        if self.warming_up:
            message = 'warming up'
        elif self.side == 'asks':
            ask_price, ask_value = self.get_ask()
            message = '%s x %s' % (str(round(ask_price, 2)), str(round(ask_value['size'], 2)))
        else:
            bid_price, bid_value = self.get_bid()
            message = '%s x %s' % (str(round(bid_value['size'], 2)), str(round(bid_price, 2)))
        return message

    def clear(self):
        """
        Reset price tree and order map
        :return: void
        """
        self.price_dict = SortedDict()
        self.order_map = dict()
        self.warming_up = True

    def create_price(self, price):
        """
        Create new node
        :param price:
        :return:
        """
        self.price_dict[price] = {'size': float(0), 'count': int(0)}

    def remove_price(self, price):
        """
        Remove node
        :param price:
        :return:
        """
        del self.price_dict[price]

    def receive(self, msg):
        """
        add incoming orders to order map
        :param msg:
        :return:
        """
        pass

    @abstractmethod
    def insert_order(self, msg):
        """
        Create new node
        :param msg:
        :return:
        """
        pass

    def _insert_orders(self, price, size, oid, product_id, side):
        """
        Used for preloading limit order book for legacy version
        of this project for GDAX only.
        :param price: order price
        :param size: order size
        :param oid: order id
        :param side: buy or sell
        :param product_id: currency ticker
        :return:
        """
        order = {
            'order_id': oid,
            'price': float(price),
            'size': float(size),
            'side': side,
            'time': dt.now(),
            'type': 'preload',
            'product_id': product_id
        }
        if order['price'] not in self.price_dict:
            self.create_price(order['price'])
        self.price_dict[order['price']]['size'] += order['size']
        self.price_dict[order['price']]['count'] += 1
        self.order_map[order['order_id']] = order

    @abstractmethod
    def match(self, msg):
        """
        Change volume of book
        :param msg:
        :return:
        """
        pass

    @abstractmethod
    def change(self, msg):
        """
        Update inventory
        :param msg:
        :return:
        """
        pass

    @abstractmethod
    def remove_order(self, msg):
        """
        Done messages result in the order being removed from map
        :param msg:
        :return:
        """
        pass

    def get_ask(self):
        """
        Best offer
        :return: inside ask
        """
        if len(self.price_dict) > 0:
            return self.price_dict.items()[0]
        else:
            return 0.0

    def get_bid(self):
        """
        Best bid
        :return: inside bid
        """
        if len(self.price_dict) > 0:
            return self.price_dict.items()[-1]
        else:
            return 0.0

    # @abstractmethod
    # def get_asks_to_list(self):
    #     pass
    #
    # @abstractmethod
    # def get_bids_to_list(self):
    #     pass
