from configurations import RECORD_DATA
from data_recorder.connector_components.book import Book


class BitfinexBook(Book):

    def __init__(self, **kwargs):
        super(BitfinexBook, self).__init__(**kwargs)

    def insert_order(self, msg: dict) -> None:
        """
        Create new node.

        :param msg: incoming new order
        :return: (void)
        """
        self.order_map[msg['order_id']] = msg

        price = msg['price']
        if price not in self.price_dict:
            self.create_price(price)

        size = abs(msg['size'])
        self.price_dict[price].add_limit(quantity=size, price=price)
        self.price_dict[price].add_quantity(quantity=size, price=price)
        self.price_dict[price].add_count()

    def match(self, msg: dict) -> None:
        """
        This method is not implemented within Bitmex's API.

        However, I've implemented it to capture order arrival flows (i.e., incoming
        market orders.) and to be consistent with the overarching design pattern.

        Note: this event handler does not impact the LOB in any other way than updating
        the number of market orders received at a given price level.

        :param msg: buy or sell transaction message from Bitfinex
        :return: (void)
        """
        price = msg.get('price', None)
        if price in self.price_dict:
            quantity = abs(msg['size'])
            self.price_dict[price].add_market(quantity=quantity, price=price)

    def change(self, msg: dict) -> None:
        """
        Update inventory.

        :param msg: order update message from Bitfinex
        :return: (void)
        """
        old_order = self.order_map[msg['order_id']]
        diff = msg['size'] - old_order['size']

        vol_change = diff != float(0)
        px_change = msg['price'] != old_order['price']

        if px_change:
            self.remove_order(old_order)
            self.insert_order(msg)

        elif vol_change:
            old_order['size'] = msg['size']
            price = old_order['price']
            self.order_map[msg['order_id']] = old_order
            self.price_dict[price].add_quantity(quantity=diff, price=price)

    def remove_order(self, msg: dict) -> None:
        """
        Done messages result in the order being removed from map.

        :param msg: remove order message from Bitfinex
        :return: (void)
        """
        msg_order_id = msg.get('order_id', None)
        if msg_order_id in self.order_map:

            old_order = self.order_map[msg_order_id]
            price = old_order['price']

            if price not in self.price_dict:
                print('remove_order: price not in msg...adj_price = {} '.format(
                    price))
                print('Incoming order: %s' % msg)
                print('Old order: %s' % old_order)

            order_size = abs(old_order.get('size', None))
            order_price = old_order.get('price', None)
            # Note: Bitfinex does not have 'canceled' message types, thus it is not
            # possible to distinguish filled orders from canceled orders with the order
            # arrival trackers.
            self.price_dict[price].add_cancel(quantity=order_size, price=order_price)
            self.price_dict[price].remove_quantity(quantity=order_size, price=order_price)
            self.price_dict[price].remove_count()

            if self.price_dict[price].count == 0:
                self.remove_price(price)

            del self.order_map[old_order['order_id']]

        elif RECORD_DATA:
            print('remove_order: order_id not found %s\n' % msg)
