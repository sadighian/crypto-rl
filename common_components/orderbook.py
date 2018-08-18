from abc import ABC, abstractmethod
from gdax_connector.gdax_book import GdaxBook
from bitfinex_connector.bitfinex_book import BitfinexBook
from common_components.database import Database
import pandas as pd


class OrderBook(ABC):

    def __init__(self, ccy, exchange):
        self.sym = ccy
        self.db = Database(ccy, exchange)
        self.bids = GdaxBook(ccy, 'bids') if exchange == 'gdax' else BitfinexBook(ccy, 'bids')
        self.asks = GdaxBook(ccy, 'asks') if exchange == 'gdax' else BitfinexBook(ccy, 'asks')

    def __str__(self):
        return '%s  |  %s' % (self.bids, self.asks)

    @abstractmethod
    def new_tick(self, msg):
        pass

    def clear_book(self):
        """
        Method to reset the limit order book
        :return: void
        """
        self.bids.clear()
        self.asks.clear()

    def render_book(self):
        """
        Convert the limit order book into a DataFrame
        :return: pandas dataframe
        """

        pd_bids = self.bids._get_bids_to_list()
        pd_asks = self.asks._get_asks_to_list()

        return pd.concat([pd_bids, pd_asks], sort=False)

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
