from data_recorder.connector_components .client import Client


class CoinbaseClient(Client):

    def __init__(self, ccy):
        super(CoinbaseClient, self).__init__(ccy, 'coinbase')

    def run(self):
        """
        Handle incoming level 3 data on a separate thread
        :return: void
        """
        while True:
            msg = self.queue.get()

            if self.book.new_tick(msg) is False:
                self.book.load_book()
                self.retry_counter += 1
                print('\n[Coinbase - %s] ...going to try and reload the order book\n' % self.sym)
                continue
