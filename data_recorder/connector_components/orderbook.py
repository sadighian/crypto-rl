from data_recorder.coinbase_connector.coinbase_book import CoinbaseBook
from data_recorder.bitfinex_connector.bitfinex_book import BitfinexBook
from data_recorder.connector_components.trade_tracker import TradeTracker
from data_recorder.database.database import Database
from configurations.configs import INCLUDE_ORDERFLOW
from abc import ABC, abstractmethod
import numpy as np


class OrderBook(ABC):

    def __init__(self, ccy, exchange):
        """
        OrderBook constructor

        :param ccy: currency symbol
        :param exchange: 'coinbase' or 'bitfinex'
        """
        self.sym = ccy
        self.db = Database(ccy, exchange)
        self.bids = CoinbaseBook(ccy, 'bids') if exchange == 'coinbase' else \
            BitfinexBook(ccy, 'bids')
        self.asks = CoinbaseBook(ccy, 'asks') if exchange == 'coinbase' else \
            BitfinexBook(ccy, 'asks')
        self.midpoint = float()
        self.buy_tracker = TradeTracker()
        self.sell_tracker = TradeTracker()

    def __str__(self):
        return '%s  ||  %s' % (self.bids, self.asks)

    @abstractmethod
    def new_tick(self, msg):
        """
        Event handler for incoming tick messages

        :param msg: incoming order or trade message
        :return:
        """
        pass

    def clear_trade_trackers(self):
        """
        Reset buy and sell trade trackers; used between LOB snapshots

        :return: (void)
        """
        self.buy_tracker.clear()
        self.sell_tracker.clear()

    def clear_book(self):
        """
        Method to reset the limit order book

        :return: (void)
        """
        self.bids.clear()
        self.asks.clear()
        print('--Cleared %s order book--' % self.sym)

    # def render_book(self):
    #     """
    #     Convert the limit order book into a DataFrame
    #     :return: pandas dataframe
    #     """
    #
    #     pd_bids = self.bids.get_bids_to_list()
    #     pd_asks = self.asks.get_asks_to_list()
    #
    #     return pd.concat([pd_bids, pd_asks], sort=False)

    def render_book(self):
        """
        Create stationary feature set for limit order book

        Source: https://arxiv.org/abs/1810.09965v1

        :return: numpy array
        """
        bid_price, bid_level = self.bids.get_bid()
        ask_price, ask_level = self.asks.get_ask()

        self.midpoint = (bid_price + ask_price) / 2.0

        bid_data = self.bids.get_bids_to_list(midpoint=self.midpoint)
        ask_data = self.asks.get_asks_to_list(midpoint=self.midpoint)

        buy_trades = np.array(self.buy_tracker.notional)
        sell_trades = np.array(self.sell_tracker.notional)
        self.clear_trade_trackers()

        if INCLUDE_ORDERFLOW:
            bid_distances, bid_notionals, bid_cancel_notionals, bid_limit_notionals, \
            bid_market_notionals = bid_data

            ask_distances, ask_notionals, ask_cancel_notionals, ask_limit_notionals, \
            ask_market_notionals = ask_data

            return np.hstack((bid_notionals,
                              ask_notionals,
                              bid_distances,
                              ask_distances,
                              buy_trades,
                              sell_trades,
                              bid_cancel_notionals,
                              ask_cancel_notionals,
                              bid_limit_notionals,
                              ask_limit_notionals,
                              bid_market_notionals,
                              ask_market_notionals))
        else:
            bid_distances, bid_notionals = bid_data

            ask_distances, ask_notionals = ask_data

            return np.hstack((bid_notionals,
                              ask_notionals,
                              bid_distances,
                              ask_distances,
                              buy_trades,
                              sell_trades))

    # def render_book(self):
    #     """
    #     Create stationary feature set for limit order book
    #
    #     Source: https://arxiv.org/abs/1810.09965v1
    #
    #     :return: numpy array
    #     """
    #     bid_price, bid_level = self.bids.get_bid()
    #     ask_price, ask_level = self.asks.get_ask()
    #
    #     self.midpoint = (bid_price + ask_price) / 2.0
    #
    #     bids = self.bids.get_bids_to_list(self.midpoint)
    #     asks = self.asks.get_asks_to_list(self.midpoint)
    #
    #     buy_trades = np.array(self.buy_tracker.notional)
    #     sell_trades = np.array(self.sell_tracker.notional)
    #
    #     self.clear_trade_trackers()
    #
    #     return np.hstack((bids, asks, buy_trades, sell_trades))

    @property
    def best_bid(self):
        """
        Get the best bid
        :return: float best bid
        """
        return self.bids.get_bid()

    @property
    def best_ask(self):
        """
        Get the best ask
        :return: float best ask
        """
        return self.asks.get_ask()

    def done_warming_up(self):
        """
        Flag to indicate if the entire Limit Order Book has been loaded
        :return: True if loaded / False if still waiting to download
        """
        return ~self.bids.warming_up & ~self.asks.warming_up
