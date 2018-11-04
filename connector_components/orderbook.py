from coinbase_connector.coinbase_book import CoinbaseBook
from bitfinex_connector.bitfinex_book import BitfinexBook
from database.database import Database
from abc import ABC, abstractmethod
import numpy as np


class OrderBook(ABC):

    def __init__(self, ccy, exchange):
        self.sym = ccy
        self.db = Database(ccy, exchange)
        self.bids = CoinbaseBook(ccy, 'bids') if exchange == 'coinbase' else BitfinexBook(ccy, 'bids')
        self.asks = CoinbaseBook(ccy, 'asks') if exchange == 'coinbase' else BitfinexBook(ccy, 'asks')
        self.midpoint = float()
        self.trade_tracker = dict({'buys': float(0),
                                   'sells': float(0)})

    def __str__(self):
        return '%s  |  %s' % (self.bids, self.asks)

    @abstractmethod
    def new_tick(self, msg):
        pass

    def clear_trades_tracker(self):
        self.trade_tracker['buys'] = float(0)
        self.trade_tracker['sells'] = float(0)

    def clear_book(self):
        """
        Method to reset the limit order book
        :return: void
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
    #     pd_bids = self.bids._get_bids_to_list()
    #     pd_asks = self.asks._get_asks_to_list()
    #
    #     return pd.concat([pd_bids, pd_asks], sort=False)

    def render_book(self):
        """
        Create stationary feature set for limit order book

        Credit: https://arxiv.org/abs/1810.09965v1

        :return: numpy array
        """
        best_bid, bid_value = self.bids.get_bid()
        best_ask, ask_value = self.asks.get_ask()
        self.midpoint = (best_bid + best_ask) / 2.0

        bids = self.bids._get_bids_to_list(self.midpoint)
        asks = self.asks._get_asks_to_list(self.midpoint)

        buy_trades = np.array(self.trade_tracker['buys'])
        sell_trades = np.array(self.trade_tracker['sells'])

        self.clear_trades_tracker()

        return np.hstack((bids, asks, buy_trades, sell_trades))

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
        return ~self.bids.warming_up & ~self.asks.warming_up
