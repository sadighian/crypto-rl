from abc import ABC, abstractmethod

import numpy as np
from sortedcontainers import SortedDict

from configurations import INCLUDE_ORDERFLOW, MAX_BOOK_ROWS
from data_recorder.connector_components.price_level import PriceLevel


class Book(ABC):
    CLEAR_MAX_ROWS = MAX_BOOK_ROWS + 45

    def __init__(self, sym: str, side: str):
        """
        Book constructor.

        :param sym: currency symbol
        :param side: 'bids' or 'asks'
        """
        self.price_dict = SortedDict()
        self.order_map = dict()
        self.side = side
        self.sym = sym
        self.warming_up = True
        # render order book using numpy for faster performance
        # LOB statistics
        self._distances = np.empty(MAX_BOOK_ROWS, dtype=np.float32)
        self._notionals = np.empty(MAX_BOOK_ROWS, dtype=np.float32)
        self._cumulative_notionals = np.empty(MAX_BOOK_ROWS, dtype=np.float32)
        # order flow arrival statistics
        self._cancel_notionals = np.empty(MAX_BOOK_ROWS, dtype=np.float32)
        self._limit_notionals = np.empty(MAX_BOOK_ROWS, dtype=np.float32)
        self._market_notionals = np.empty(MAX_BOOK_ROWS, dtype=np.float32)

    def __str__(self):
        if self.warming_up:
            message = 'warming up'
        elif self.side == 'asks':
            ask_price, ask_price_level = self.get_ask()
            message = '{:>8,.2f} x {:>9,.0f} | {:>9,.0f}'.format(
                ask_price,
                # ask_price_level.quantity,
                ask_price_level.notional,
                ask_price_level.limit_notional - ask_price_level.cancel_notional +
                ask_price_level.market_notional,
            )
        else:
            bid_price, bid_price_level = self.get_bid()
            message = '{:>9,.0f} | {:>9,.0f} x {:>8,.2f}'.format(
                bid_price_level.limit_notional - bid_price_level.cancel_notional +
                bid_price_level.market_notional,
                # bid_price_level.quantity,
                bid_price_level.notional,
                bid_price
            )
        return message

    def clear(self) -> None:
        """
        Reset price tree and order map.

        :return: void
        """
        self.price_dict = SortedDict()
        self.order_map = dict()
        self.warming_up = True

    def create_price(self, price: float) -> None:
        """
        Create new node.

        :param price: price level to create in LOB
        :return:
        """
        self.price_dict[price] = PriceLevel(price=price, quantity=0.)

    def remove_price(self, price: float) -> None:
        """
        Remove node.

        :param price: price level to remove from LOB
        :return:
        """
        del self.price_dict[price]

    def receive(self, msg) -> None:
        """
        add incoming orders to order map.

        :param msg: new message received by exchange
        :return:
        """
        pass

    @abstractmethod
    def insert_order(self, msg: dict) -> None:
        """
        Insert order into existing node.

        :param msg: new limit order
        :return:
        """
        pass

    @abstractmethod
    def match(self, msg: dict) -> None:
        """
        Change volume of book; used with time and sales data.

        :param msg: buy or sell execution
        :return:
        """
        pass

    @abstractmethod
    def change(self, msg: dict) -> None:
        """
        Update inventory.

        :param msg: update order request
        :return:
        """
        pass

    @abstractmethod
    def remove_order(self, msg: dict) -> None:
        """
        Done messages result in the order being removed from map

        :param msg:
        :return:
        """
        pass

    def get_ask(self) -> (float, PriceLevel):
        """
        Best offer

        :return: (float) inside ask, (PriceLevel) ask size and number of orders
        """
        if len(self.price_dict) > 0:
            return self.price_dict.items()[0]
        else:
            return 0.0, PriceLevel(price=0., quantity=0.)

    def get_bid(self) -> (float, PriceLevel):
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
                              level_number: int) -> float:
        """
        Iterate through LOB and pass cumulative_notional recursively. Implemented in
        numpy for speed.

        :param price: raw price of current price-level 'i' in LOB
        :param midpoint: midpoint price
        :param level: PriceLevel object which stores notionals, quantity, etc.
        :param cumulative_notional: cumulative notional value from walking the LOB
        :param level_number: level 'i' in LOB
        :return: current cumulative notional at level 'i'
        """
        # order book stats
        self._distances[level_number] = (price / midpoint) - 1.
        self._notionals[level_number] = level.notional
        cumulative_notional += level.notional
        self._cumulative_notionals[level_number] = cumulative_notional  # <- Note: not
        # implemented yet
        return cumulative_notional

    def _add_to_order_flow_trackers(self, level: PriceLevel, level_number: int) -> None:
        """
        Iterate through LOB to capture order arrival statistics. Implemented in numpy
        for speed.

        :param level: PriceLevel object which stores notionals, quantity, etc.
        :param level_number: level 'i' in LOB
        """
        # order flow arrival statistics
        self._cancel_notionals[level_number] = level.cancel_notional
        self._limit_notionals[level_number] = level.limit_notional
        self._market_notionals[level_number] = level.market_notional

    def get_asks_to_list(self, midpoint: float) -> tuple:
        """
        Walk the LOB to derive:
            1.) price-level distance to midpoint
            2.) notional value of each price-level
            **Optional**
            3.) notional values accumulated since last snapshot for cancel, market,
                and limit orders

        :param midpoint: current midpoint
        :return: tuple containing derived LOB feature set
        """
        cumulative_notional = 0.
        book_rows_to_clear = Book.CLEAR_MAX_ROWS if INCLUDE_ORDERFLOW else MAX_BOOK_ROWS

        for i, (price, level) in enumerate(self.price_dict.items()[:book_rows_to_clear]):
            if i < MAX_BOOK_ROWS:
                # only include price levels that are within the specified range
                cumulative_notional = self._add_to_book_trackers(
                    price=price, midpoint=midpoint, level=level,
                    cumulative_notional=cumulative_notional, level_number=i
                )
                self._add_to_order_flow_trackers(level=level, level_number=i)
            # clear the trackers on nearby price levels in case of price jumps,
            # but do not clear all the price levels in the LOB to save time.
            level.clear_trackers()  # to prevent orders close to the top-n from

        # append all the data points together
        book_data = (self._distances, self._notionals,)

        # include order flow arrival statistics
        if INCLUDE_ORDERFLOW:
            book_data += (self._cancel_notionals, self._limit_notionals,
                          self._market_notionals,)

        return book_data

    def get_bids_to_list(self, midpoint: float) -> tuple:
        """
        Walk the LOB to derive:
            1.) price-level distance to midpoint
            2.) notional value of each price-level
            **Optional**
            3.) notional values accumulated since last snapshot for cancel, market,
                and limit orders

        Note: currently configured to return all data slices in ascending order
            (not mirroring)

        :param midpoint: current midpoint
        :return: tuple containing derived LOB feature set
        """
        cumulative_notional = 0.
        book_rows_to_clear = Book.CLEAR_MAX_ROWS if INCLUDE_ORDERFLOW else MAX_BOOK_ROWS

        for i, (price, level) in enumerate(
                reversed(self.price_dict.items()[-book_rows_to_clear:])):

            if i < MAX_BOOK_ROWS:
                # only include price levels that are within the specified range
                cumulative_notional = self._add_to_book_trackers(
                    # price, midpoint, level, cumulative_notional, MAX_BOOK_ROWS - i - 1
                    price=price, midpoint=midpoint, level=level,
                    cumulative_notional=cumulative_notional, level_number=i
                )
                # self._add_to_order_flow_trackers(level, MAX_BOOK_ROWS - i - 1)
                self._add_to_order_flow_trackers(level=level, level_number=i)
            # clear the trackers on nearby price levels in case of price jumps,
            # but do not clear all the price levels in the LOB to save time.
            level.clear_trackers()  # to prevent orders close to the top-n from

        # append all the data points together
        book_data = (self._distances, self._notionals,)

        # include order flow arrival statistics
        if INCLUDE_ORDERFLOW:
            book_data += (self._cancel_notionals, self._limit_notionals,
                          self._market_notionals,)

        return book_data
