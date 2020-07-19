from abc import ABC, abstractmethod

import numpy as np

from configurations import INCLUDE_ORDERFLOW, LOGGER, MAX_BOOK_ROWS
from data_recorder.bitfinex_connector.bitfinex_book import BitfinexBook
from data_recorder.coinbase_connector.coinbase_book import CoinbaseBook
from data_recorder.connector_components.trade_tracker import TradeTracker
from data_recorder.database.database import Database

BOOK_BY_EXCHANGE = dict(coinbase=CoinbaseBook, bitfinex=BitfinexBook)


class OrderBook(ABC):

    def __init__(self, sym: str, exchange: str):
        """
        OrderBook constructor.

        :param sym: instrument name
        :param exchange: 'coinbase' or 'bitfinex' or 'bitmex'
        """
        self.sym = sym
        self.db = Database(sym=sym, exchange=exchange)
        self.db.init_db_connection()
        self.bids = BOOK_BY_EXCHANGE[exchange](sym=sym, side='bids')
        self.asks = BOOK_BY_EXCHANGE[exchange](sym=sym, side='asks')
        self.exchange = exchange
        self.midpoint = float()
        self.spread = float()
        self.buy_tracker = TradeTracker()
        self.sell_tracker = TradeTracker()
        self.last_tick_time = None

    def __str__(self):
        return '{:>8,.0f} <> {}  ||  {} <> {:>8,.0f}'.format(
            self.sell_tracker.notional, self.bids, self.asks, self.buy_tracker.notional)

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
        LOGGER.info(f"{self.sym}'s order book cleared.")

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

        return np.hstack((self.midpoint, self.spread, buy_trades, sell_trades,
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

        for side in ['bids', 'asks']:
            for feature in feature_types:
                for row in range(MAX_BOOK_ROWS):
                    feature_names.append(f"{side}_{feature}_{row}")

        LOGGER.info(f"render_feature_names() has {len(feature_names)} features")

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
