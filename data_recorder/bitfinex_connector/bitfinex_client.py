import json

import websockets

from configurations import BITFINEX_ENDPOINT, LOGGER
from data_recorder.bitfinex_connector.bitfinex_orderbook import BitfinexOrderBook
from data_recorder.connector_components.client import Client


class BitfinexClient(Client):

    def __init__(self, **kwargs):
        """
        Constructor for Bitfinex Client.

        :param sym: Instrument or cryptocurrency pair name
        :param db_queue: (optional) queue to connect to the database process
        """
        super(BitfinexClient, self).__init__(exchange='bitfinex', **kwargs)
        self.request = json.dumps({
            "event": "subscribe",
            "channel": "book",
            "prec": "R0",
            "freq": "F0",
            "symbol": self.sym,
            "len": "100"
        })
        self.request_unsubscribe = None
        self.trades_request = json.dumps({
            "event": "subscribe",
            "channel": "trades",
            "symbol": self.sym
        })
        self.book = BitfinexOrderBook(sym=self.sym)
        self.ws_endpoint = BITFINEX_ENDPOINT

    async def unsubscribe(self) -> None:
        """
        Send unsubscribe requests to exchange.

        Returns
        -------

        """
        for channel in self.book.channel_id:
            self.request_unsubscribe = {
                "event": "unsubscribe",
                "chanId": channel
            }
            await super(BitfinexClient, self).unsubscribe()

    def run(self) -> None:
        """
        Handle incoming level 3 data on a separate thread or process.

        Returns
        -------

        """
        super(BitfinexClient, self).run()
        while True:
            msg = self.queue.get()

            if self.book.new_tick(msg) is False:
                self.retry_counter += 1
                self.book.clear_book()
                LOGGER.info('\n[%s - %s] ...going to try and reload the order book\n'
                            % (self.exchange.upper(), self.sym))
                raise websockets.ConnectionClosed(10001, '%s: no explanation' %
                                                  self.exchange.upper())
                # raise an exception to invoke reconnecting
