from abc import ABC, abstractmethod
from sortedcontainers import SortedDict
import numpy as np
from configurations.configs import MAX_BOOK_ROWS


class Book(ABC):

    def __init__(self, sym, side):
        self.price_dict = SortedDict()
        self.order_map = dict()
        self.side = side
        self.sym = sym
        self.warming_up = True
        # this value needs to be set within the orderbook class

    def __str__(self):
        if self.warming_up:
            message = 'warming up'
        elif self.side == 'asks':
            ask_price, ask_value = self.get_ask()
            message = '%s x %s' % \
                      (str(round(ask_price, 3)), str(round(ask_value['size'], 2)))
        else:
            bid_price, bid_value = self.get_bid()
            message = '%s x %s' % \
                      (str(round(bid_value['size'], 2)), str(round(bid_price, 3)))
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
        :return: (float) inside ask, (dict) ask size and number of orders
        """
        if len(self.price_dict) > 0:
            return self.price_dict.items()[0]
        else:
            return 0.0, dict({'size': 0.0, 'count': 0})

    def get_bid(self):
        """
        Best bid
        :return: (float) inside bid, (dict) bid size and number of orders
        """
        if len(self.price_dict) > 0:
            return self.price_dict.items()[-1]
        else:
            return 0.0, dict({'size': 0.0, 'count': 0})

    def _get_asks_to_list(self, midpoint):
        """
        Source: https://arxiv.org/abs/1810.09965v1
        """
        notionals = list()
        cumulative_notional = float(0.0)
        distances = list()

        for k, v in self.price_dict.items()[:MAX_BOOK_ROWS]:
            distances.append(k - midpoint)
            cumulative_notional += float(k * v['size'])
            notionals.append(cumulative_notional)

        return np.array((notionals + distances))

    def _get_bids_to_list(self, midpoint):
        """
        Source: https://arxiv.org/abs/1810.09965v1
        """
        notionals = list()
        cumulative_notional = float(0.0)
        distances = list()

        for k, v in reversed(self.price_dict.items()[-MAX_BOOK_ROWS:]):
            distances.append(midpoint - k)
            cumulative_notional += float(k * v['size'])
            notionals.append(cumulative_notional)

        return np.array((notionals + distances))

    # def _get_asks_to_list(self):
    #     """
    #     Transform order book to dictionary with 3 lists:
    #         1- ask prices
    #         2- cumulative ask volume at a given price
    #         3- number of orders resting at a given price
    #     :return: dictionary
    #     """
    #     prices, sizes, counts = list(), list(), list()
    #     for k, v in self.price_dict.items()[:MAX_BOOK_ROWS]:
    #         prices.append(k)
    #         sizes.append(v['size'])
    #         counts.append(v['count'])
    #
    #     # return dict({'prices': prices, 'sizes': sizes, 'counts': counts})
    #     return list(np.hstack((prices, sizes, counts)))
    #
    # def _get_bids_to_list(self):
    #     """
    #     Transform order book to dictionary with 3 lists:
    #         1- bid prices
    #         2- cumulative bid volume at a given price
    #         3- number of orders resting at a given price
    #     :return: dictionary
    #     """
    #     prices, sizes, counts = list(), list(), list()
    #     for k, v in reversed(self.price_dict.items()[-MAX_BOOK_ROWS:]):
    #         prices.append(k)
    #         sizes.append(v['size'])
    #         counts.append(v['count'])
    #
    #     # return dict({'prices': prices, 'sizes': sizes, 'counts': counts})
    #     return list(np.hstack((prices, sizes, counts)))
