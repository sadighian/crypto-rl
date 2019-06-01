from data_recorder.connector_components.client import Client
import websockets


class BitfinexClient(Client):

    def __init__(self, ccy):
        super(BitfinexClient, self).__init__(ccy, 'bitfinex')

    def run(self):
        """
        Handle incoming level 3 data on a separate thread
        :return:
        """
        while True:
            msg = self.queue.get()

            if self.book.new_tick(msg) is False:
                self.retry_counter += 1
                self.book.clear_book()
                print('\n[Bitfinex - %s] ...going to try and reload the order book\n'
                      % self.sym)
                raise websockets.ConnectionClosed(1006, 'no reason')
                # raise an exception to invoke reconnecting
