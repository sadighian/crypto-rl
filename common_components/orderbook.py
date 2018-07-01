from abc import ABC, abstractmethod
from gdax_connector.gdax_book import GdaxBook
from bitfinex_connector.bitfinex_book import BitfinexBook
from datetime import datetime as dt


class OrderBook(ABC):

    def __init__(self, ccy, exchange):
        self.sym = ccy
        self.bids = GdaxBook(ccy, 'bids') if exchange == 'gdax' else BitfinexBook(ccy, 'bids')
        self.asks = GdaxBook(ccy, 'asks') if exchange == 'gdax' else BitfinexBook(ccy, 'asks')
        self.trades = dict({
            'upticks': {
                'size': float(0),
                'count': int(0)
            },
            'downticks': {
                'size': float(0),
                'count': int(0)
            }
        })

    def __str__(self):
        return '%s  |  %s' % (self.bids, self.asks)

    @abstractmethod
    def new_tick(self, msg):
        pass

    def _reset_trades_tracker(self):
        self.trades = dict({
            'upticks': {
                'size': float(0),
                'count': int(0)
            },
            'downticks': {
                'size': float(0),
                'count': int(0)
            }
        })

    def clear_book(self):
        """
        Method to reset the limit order book
        :return: void
        """
        self.bids.clear()
        self.asks.clear()

    def render_book(self):
        """
        Convert the limit order book into a dictionary
        :return: dictionary
        """
        book = dict({
            'bids': self.bids._get_bids_to_list(),
            'asks': self.asks._get_asks_to_list(),
            'upticks': self._get_trades_tracker['upticks'],
            'downticks': self._get_trades_tracker['downticks'],
            'time': dt.now()
        })
        self._reset_trades_tracker()
        return book

    def render_price(self, side, reference):
        """
        Estimate market order slippage
        :param side: bids or asks
        :param reference: NBBO
        :return: float distance from NBBO
        """
        if side == 'bids':
            return round(self.bids._do_next_price('bids', reference), 2)
        else:
            return round(self.asks._do_next_price('asks', reference), 2)

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

    @property
    def _get_trades_tracker(self):
        return self.trades
