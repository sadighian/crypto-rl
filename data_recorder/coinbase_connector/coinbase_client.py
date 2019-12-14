from data_recorder.connector_components.client import Client
from data_recorder.coinbase_connector.coinbase_orderbook import CoinbaseOrderBook
from configurations import COINBASE_ENDPOINT, LOGGER
import json


class CoinbaseClient(Client):

    def __init__(self, ccy: str):
        """
        Constructor for Coinbase Client.

        Parameters
        ----------
        ccy : str
            Name of instrument or cryptocurrency pair.
        """
        super(CoinbaseClient, self).__init__(ccy=ccy, exchange='coinbase')
        self.request = json.dumps(dict(type='subscribe',
                                       product_ids=[self.sym],
                                       channels=['full']))
        self.request_unsubscribe = json.dumps(dict(type='unsubscribe',
                                                   product_ids=[self.sym],
                                                   channels=['full']))
        self.book = CoinbaseOrderBook(sym=self.sym)
        self.trades_request = None
        self.ws_endpoint = COINBASE_ENDPOINT

    def run(self):
        """
        Handle incoming level 3 data on a separate thread or process.

        Returns
        -------

        """
        super(CoinbaseClient, self).run()
        while True:
            msg = self.queue.get()

            if self.book.new_tick(msg) is False:
                self.book.load_book()
                self.retry_counter += 1
                LOGGER.info('\n[%s - %s] ...going to try and reload the order '
                            'book\n' % (self.exchange.upper(), self.sym))
                continue
