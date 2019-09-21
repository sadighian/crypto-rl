from data_recorder.connector_components.book import Book, round_price
from configurations.configs import RECORD_DATA


class BitfinexBook(Book):

    def __init__(self, sym, side):
        super(BitfinexBook, self).__init__(sym, side)

    def insert_order(self, msg):
        """
        Create new node
        :param msg: incoming new order
        :return: (void)
        """
        self.order_map[msg['order_id']] = msg
        price = msg.get('price', None)
        adj_price = round_price(price)
        if adj_price not in self.price_dict:
            self.create_price(adj_price)

        quantity = abs(msg['size'])
        self.price_dict[adj_price].add_limit(quantity=quantity, price=price)
        self.price_dict[adj_price].add_quantity(quantity=quantity, price=price)
        self.price_dict[adj_price].add_count()

    def match(self, msg):
        """
        This method is not implemented within Bitfinex's API.

        However, I've implemented it to capture order arrival flows (i.e., incoming
        market orders.) and to be consistent with the overarching design pattern.

        Note: this event handler does not impact the LOB in any other way than updating
        the number of market orders received at a given price level.

        :param msg: buy or sell transaction message from Bitfinex
        :return: (void)
        """
        price = msg.get('price', None)
        adj_price = round_price(price)
        if adj_price in self.price_dict:
            self.price_dict[adj_price].add_market(quantity=msg['size'],
                                                  price=price)

    def change(self, msg):
        """
        Update inventory
        :param msg: order update message from Bitfinex
        :return: (void)
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
            adj_price = round_price(old_order['price'])
            self.order_map[msg['order_id']] = old_order
            self.price_dict[adj_price].remove_quantity(quantity=diff,
                                                       price=old_order['price'])
            assert px_change is False

    def remove_order(self, msg):
        """
        Done messages result in the order being removed from map
        :param msg: remove order message from Bitfinex
        :return: (void)
        """
        msg_order_id = msg.get('order_id', None)
        if msg_order_id in self.order_map:

            old_order = self.order_map[msg_order_id]
            adj_price = round_price(old_order['price'])

            if adj_price not in self.price_dict:
                print('remove_order: price not in msg...adj_price = {} '.format(
                    adj_price))
                print('Incoming order: %s' % msg)
                print('Old order: %s' % old_order)

            order_size = abs(old_order.get('size', None))
            order_price = old_order.get('price', None)
            # Note: Bitfinex does not have 'canceled' message types, thus it is not
            # possible to distinguish filled orders from canceled orders with the order
            # arrival trackers.
            self.price_dict[adj_price].add_cancel(quantity=order_size,
                                                  price=order_price)
            self.price_dict[adj_price].remove_quantity(quantity=order_size,
                                                       price=order_price)
            self.price_dict[adj_price].remove_count()

            if self.price_dict[adj_price].count == 0:
                self.remove_price(adj_price)

            del self.order_map[old_order['order_id']]

        elif RECORD_DATA:
            print('remove_order: order_id not found %s\n' % msg)
