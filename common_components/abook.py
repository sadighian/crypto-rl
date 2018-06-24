from abc import ABC, abstractmethod
from gdax_connector.diction import Diction as GdaxDiction
from bitfinex_connector.diction import Diction as BitfinexDiction
from datetime import datetime as dt


class ABook(ABC):

    def __init__(self, ccy, exchange):
        self.sym = ccy
        self.bids = GdaxDiction(ccy, 'bids') if exchange == 'gdax' else BitfinexDiction(ccy, 'bids')
        self.asks = GdaxDiction(ccy, 'asks') if exchange == 'gdax' else BitfinexDiction(ccy, 'asks')
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

    @property
    def _get_trades_tracker(self):
        return self.trades


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
            'bids': self.bids.get_bids_to_list(),
            'asks': self.asks.get_asks_to_list(),
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
            return round(self.bids.do_next_price('bids', reference), 2)
        else:
            return round(self.asks.do_next_price('asks', reference), 2)

    def best_bid(self):
        """
        Get the best bid
        :return: float best bid
        """
        return self.bids.get_bid()

    def best_ask(self):
        """
        Get the best ask
        :return: float best ask
        """
        return self.asks.get_ask()

    @abstractmethod
    def new_tick(self, msg):
        pass

