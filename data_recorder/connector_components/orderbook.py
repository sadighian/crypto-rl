from data_recorder.coinbase_connector.coinbase_book import CoinbaseBook
from data_recorder.bitfinex_connector.bitfinex_book import BitfinexBook
from data_recorder.connector_components.trade_tracker import TradeTracker
from data_recorder.database.database import Database
from configurations import INCLUDE_ORDERFLOW, LOGGER, MAX_BOOK_ROWS
from abc import ABC, abstractmethod
from typing import Type, Union
import numpy as np


def get_orderbook(name: str) -> Type[Union[CoinbaseBook, BitfinexBook]]:
    """
    Helper function to get the order book for a given exchange name.

    :param name: name of the exchange
    :return:
    """
    return dict(coinbase=CoinbaseBook, bitfinex=BitfinexBook)[name]


class OrderBook(ABC):

    def __init__(self, ccy: str, exchange: str):
        """
        OrderBook constructor.

        :param ccy: currency symbol
        :param exchange: 'coinbase' or 'bitfinex'
        """
        self.sym = ccy
        self.db = Database(sym=ccy, exchange=exchange)
        self.db.init_db_connection()
        self.bids = get_orderbook(name=exchange)(sym=ccy, side='bids')
        self.asks = get_orderbook(name=exchange)(sym=ccy, side='asks')
        self.exchange = exchange
        self.midpoint = float()
        self.spread = float()
        self.buy_tracker = TradeTracker()
        self.sell_tracker = TradeTracker()
        self.last_tick_time = None

    def __str__(self):
        return '%s  ||  %s' % (self.bids, self.asks)

    @abstractmethod
    def new_tick(self, msg: dict) -> bool:
        """
        Event handler for incoming tick messages.

        :param msg: incoming order or trade message
        :return: FALSE if reconnection to WebSocket is needed, else TRUE if good
        """
        return True

    def clear_trade_trackers(self) -> None:
        """
        Reset buy and sell trade trackers; used between LOB snapshots.

        :return: (void)
        """
        self.buy_tracker.clear()
        self.sell_tracker.clear()

    def clear_book(self) -> None:
        """
        Method to reset the limit order book.

        :return: (void)
        """
        self.bids.clear()  # warming_up flag reset in `Position` class
        self.asks.clear()  # warming_up flag reset in `Position` class
        self.last_tick_time = None
        LOGGER.info("{}'s order book cleared.".format(self.sym))

    def render_book(self) -> np.ndarray:
        """
        Create stationary feature set for limit order book.

        :return: LOB feature set
        """
        # get price levels of LOB
        bid_price, bid_level = self.bids.get_bid()
        ask_price, ask_level = self.asks.get_ask()

        # derive midpoint price and spread from bid and ask data
        self.midpoint = (ask_price + bid_price) / 2.0
        self.spread = round(ask_price - bid_price, 4)  # round to clean float rounding

        # transform raw LOB data into stationary feature set
        bid_data = self.bids.get_bids_to_list(midpoint=self.midpoint)
        ask_data = self.asks.get_asks_to_list(midpoint=self.midpoint)

        # convert buy and sell trade notional values to an array
        buy_trades = np.array(self.buy_tracker.notional)
        sell_trades = np.array(self.sell_tracker.notional)

        # reset trackers after each LOB render
        self.clear_trade_trackers()

        return np.hstack((self.midpoint, self.spread,
                          buy_trades, sell_trades,
                          *bid_data, *ask_data))

    @staticmethod
    def render_lob_feature_names(include_orderflow: bool = INCLUDE_ORDERFLOW) -> list:
        """
        Get the column names for the LOB render features.

        :param include_orderflow: if TRUE, order flow imbalance stats are included in set
        :return: list containing features names
        """
        feature_names = list()

        feature_names.append('midpoint')
        feature_names.append('spread')
        feature_names.append('buys')
        feature_names.append('sells')

        feature_types = ['distance', 'notional']
        if include_orderflow:
            feature_types += ['cancel_notional', 'limit_notional', 'market_notional']

        for side in ['bid', 'ask']:
            for feature in feature_types:
                for row in range(MAX_BOOK_ROWS):
                    feature_names.append("{}_{}_{}".format(side, feature, row))

        LOGGER.info("render_feature_names() has {} features".format(len(feature_names)))

        return feature_names

    @property
    def best_bid(self) -> float:
        """
        Get the best bid.

        :return: float best bid
        """
        return self.bids.get_bid()

    @property
    def best_ask(self) -> float:
        """
        Get the best ask.

        :return: float best ask
        """
        return self.asks.get_ask()

    @property
    def done_warming_up(self) -> bool:
        """
        Flag to indicate if the entire Limit Order Book has been loaded.

        :return: True if loaded / False if still waiting to download
        """
        return self.bids.warming_up is False & self.asks.warming_up is False
