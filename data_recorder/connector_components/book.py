from abc import ABC, abstractmethod
from sortedcontainers import SortedDict
import numpy as np
from configurations.configs import MAX_BOOK_ROWS, INCLUDE_ORDERFLOW, BOOK_DIGITS, \
    AGGREGATE
from data_recorder.connector_components.price_level import PriceLevel


class Book(ABC):
    CLEAR_MAX_ROWS = MAX_BOOK_ROWS + 20

    def __init__(self, sym, side):
        """
        Book constructor
        :param sym: currency symbol
        :param side: 'bids' or 'asks'
        """
        self.price_dict = SortedDict()
        self.order_map = dict()
        self.side = side
        self.sym = sym
        self.warming_up = True
        # render order book using numpy for faster performance
        self._distances = np.empty(MAX_BOOK_ROWS, dtype=np.float32)
        self._notionals = np.empty(MAX_BOOK_ROWS, dtype=np.float32)
        self._cancel_notionals = np.empty(MAX_BOOK_ROWS, dtype=np.float32)
        self._limit_notionals = np.empty(MAX_BOOK_ROWS, dtype=np.float32)
        self._market_notionals = np.empty(MAX_BOOK_ROWS, dtype=np.float32)

    def __str__(self):
        if self.warming_up:
            message = 'warming up'
        elif self.side == 'asks':
            ask_price, ask_price_level = self.get_ask()
            message = '{:.3f} x {:.4f} | {:.1f} | {:.1f} | {:.1f}'.format(
                ask_price,
                ask_price_level.quantity,
                ask_price_level.cancel_notional,
                ask_price_level.limit_notional,
                ask_price_level.market_notional)
        else:
            bid_price, bid_price_level = self.get_bid()
            message = '{:.1f} | {:.1f} | {:.1f} | {:.4f} x {:.3f}'.format(
                bid_price_level.market_notional,
                bid_price_level.limit_notional,
                bid_price_level.cancel_notional,
                bid_price_level.quantity, bid_price)
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
        self.price_dict[price] = PriceLevel(price=price, quantity=0.)

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
        :return: (float) inside ask, (PriceLevel) ask size and number of orders
        """
        if len(self.price_dict) > 0:
            return self.price_dict.items()[0]
        else:
            return 0.0, PriceLevel(price=0., quantity=0.)

    def get_bid(self):
        """
        Best bid
        :return: (float) inside bid, (PriceLevel) bid size and number of orders
        """
        if len(self.price_dict) > 0:
            return self.price_dict.items()[-1]
        else:
            return 0.0, PriceLevel(price=0., quantity=0.)

    def _add_to_book_trackers(self,
                              price: float,
                              midpoint: float,
                              level: PriceLevel,
                              cumulative_notional: float,
                              level_number: int):
        # order book metrics
        self._distances[level_number] = (price / midpoint) - 1.
        cumulative_notional += level.notional
        self._notionals[level_number] = cumulative_notional
        return cumulative_notional

    def _add_to_order_flow_trackers(self, level: PriceLevel, level_number: int):
        # order flow metrics
        self._cancel_notionals[level_number] = level.cancel_notional
        self._limit_notionals[level_number] = level.limit_notional
        self._market_notionals[level_number] = level.market_notional

    def get_asks_to_list(self, midpoint: float):
        """
        Inspired by: https://arxiv.org/abs/1810.09965v1
        """
        cumulative_notional = 0.

        if INCLUDE_ORDERFLOW:
            for i, (price, level) in enumerate(
                    self.price_dict.items()[:Book.CLEAR_MAX_ROWS]):
                if i < MAX_BOOK_ROWS:
                    cumulative_notional = self._add_to_book_trackers(
                        price, midpoint, level, cumulative_notional, i
                    )
                    self._add_to_order_flow_trackers(level, i)
                level.clear_trackers()  # to prevent orders close to the top-n from
                # entering the top-n with stale metrics
            return (
                self._distances,
                self._notionals,
                self._cancel_notionals,
                self._limit_notionals,
                self._market_notionals,
            )
        else:
            for i, (price, level) in enumerate(self.price_dict.items()[:MAX_BOOK_ROWS]):
                cumulative_notional = self._add_to_book_trackers(
                        price, midpoint, level, cumulative_notional, i
                )
            return (
                self._distances,
                self._notionals,
            )

    def get_bids_to_list(self, midpoint):
        """
        Source: https://arxiv.org/abs/1810.09965v1
        """
        cumulative_notional = 0.

        if INCLUDE_ORDERFLOW:
            for i, (price, level) in enumerate(reversed(
                    self.price_dict.items()[-Book.CLEAR_MAX_ROWS:])):
                if i < MAX_BOOK_ROWS:
                    cumulative_notional = self._add_to_book_trackers(
                        price, midpoint, level, cumulative_notional, MAX_BOOK_ROWS - i - 1
                    )
                    self._add_to_order_flow_trackers(level, MAX_BOOK_ROWS - i - 1)
                level.clear_trackers()  # to prevent orders close to the top-n from
                # entering the top-n with stale metrics
            return (
                self._distances,
                self._notionals,
                self._cancel_notionals,
                self._limit_notionals,
                self._market_notionals,
            )
        else:  # Do NOT include order arrival flow metrics
            for i, (price, level) in enumerate(reversed(
                    self.price_dict.items()[-MAX_BOOK_ROWS:])):
                cumulative_notional = self._add_to_book_trackers(
                        price, midpoint, level, cumulative_notional, MAX_BOOK_ROWS - i - 1
                )
            return (
                self._distances,
                self._notionals,
            )

    # def get_asks_to_list(self):
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
    # def get_bids_to_list(self):
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


def round_price(price=100., digits=BOOK_DIGITS, round_prices=AGGREGATE):
    return round(price, digits) if round_prices else price
