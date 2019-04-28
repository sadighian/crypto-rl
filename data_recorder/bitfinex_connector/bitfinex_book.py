from data_recorder.connector_components .book import Book
from configurations.configs import RECORD_DATA


class BitfinexBook(Book):

    def __init__(self, sym, side):
        super(BitfinexBook, self).__init__(sym, side)

    def insert_order(self, msg):
        """
        Create new node
        :param msg: incoming new order
        :return:
        """
        self.order_map[msg['order_id']] = msg

        if msg['price'] not in self.price_dict:
            self.create_price(msg['price'])

        self.price_dict[msg['price']]['size'] += abs(msg['size'])
        self.price_dict[msg['price']]['count'] += 1

    def match(self, msg):
        """
        This method is not implemented for Bitfinex
        :param msg:
        :return:
        """
        pass

    def change(self, msg):
        """
        Update inventory
        :param msg:
        :return:
        """
        old_order = self.order_map[msg['order_id']]
        diff = msg['size'] - old_order['size']

        vol_change = diff != float(0)
        px_change = old_order['price'] != msg['price']

        if px_change:
            self.remove_order(old_order)
            old_order['price'] = msg['price']
            if vol_change:
                old_order['size'] = msg['size']
            self.insert_order(old_order)

        elif vol_change:
            old_order['size'] = msg['size']
            self.order_map[msg['order_id']] = old_order
            self.price_dict[msg['price']]['size'] += diff
            assert px_change is False

    def remove_order(self, msg):
        """
        Done messages result in the order being removed from map
        :param msg:
        :return:
        """
        if msg['order_id'] in self.order_map:

            old_order = self.order_map[msg['order_id']]

            if old_order['price'] not in self.price_dict:
                print('remove_order: price not in msg...')
                print('Incoming order: %s' % msg)
                print('Old order: %s' % old_order)

            self.price_dict[old_order['price']]['size'] -= abs(old_order['size'])
            self.price_dict[old_order['price']]['count'] -= 1

            if self.price_dict[old_order['price']]['count'] == 0:
                self.remove_price(old_order['price'])

            del self.order_map[old_order['order_id']]

        elif RECORD_DATA:
            print('remove_order: order_id not found %s\n' % msg)
