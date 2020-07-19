from configurations import LOGGER, RECORD_DATA
from data_recorder.connector_components.book import Book


class CoinbaseBook(Book):

    def __init__(self, **kwargs):
        """
        Coinbase Book constructor.
        """
        super(CoinbaseBook, self).__init__(**kwargs)

    def insert_order(self, msg: dict) -> None:
        """
        Create new node.

        :param msg: incoming order message
        """
        msg_order_id = msg.get('order_id', None)
        if msg_order_id not in self.order_map:
            order = {
                'order_id': msg_order_id,
                'price': float(msg['price']),
                'size': float(msg.get('size') or msg['remaining_size']),
                'side': msg['side'],
                'time': msg['time'],
                'type': msg['type'],
                'product_id': msg['product_id']
            }
            self.order_map[order['order_id']] = order
            price = order.get('price', None)
            size = order.get('size', None)

            if price not in self.price_dict:
                self.create_price(price)

            self.price_dict[price].add_limit(quantity=size, price=price)
            self.price_dict[price].add_quantity(quantity=size, price=price)
            self.price_dict[price].add_count()

    def match(self, msg: dict) -> None:
        """
        Change volume of book.

        :param msg: incoming order message
        """
        msg_order_id = msg.get('maker_order_id', None)
        if msg_order_id in self.order_map:
            old_order = self.order_map[msg_order_id]
            order = {
                'order_id': msg_order_id,
                'price': float(msg['price']),
                'size': float(msg['size']),
                'side': msg['side'],
                'time': msg['time'],
                'type': msg['type'],
                'product_id': msg['product_id']
            }
            price = order['price']
            if price in self.price_dict:
                remove_size = order['size']
                remaining_size = old_order['size'] - remove_size
                order['size'] = remaining_size
                self.order_map[old_order['order_id']] = order
                old_order_price = old_order.get('price', None)
                self.price_dict[price].add_market(quantity=remove_size,
                                                  price=old_order_price)
                self.price_dict[price].remove_quantity(quantity=remove_size,
                                                       price=old_order_price)
            else:
                LOGGER.info('\nmatch: price not in tree already [%s]\n' % msg)
        elif RECORD_DATA:
            LOGGER.warn('\n%s match: order id cannot be found for %s\n' % (self.sym, msg))

    def change(self, msg: dict) -> None:
        """
        Update inventory.

        :param msg: incoming order message
        """
        if 'price' in msg:
            msg_order_id = msg.get('order_id', None)
            if msg_order_id in self.order_map:
                old_order = self.order_map[msg_order_id]
                new_size = float(msg['new_size'])
                old_size = old_order['size']
                diff = old_size - new_size
                old_order['size'] = new_size
                self.order_map[old_order['order_id']] = old_order
                old_order_price = old_order.get('price', None)
                self.price_dict[old_order_price].remove_quantity(quantity=diff,
                                                                 price=old_order_price)
            elif RECORD_DATA:
                LOGGER.info('\n%s change: missing order_ID [%s] from order_map\n' %
                            (self.sym, msg))

    def remove_order(self, msg: dict) -> None:
        """
        Done messages result in the order being removed from map.

        :param msg: incoming order message
        """
        msg_order_id = msg.get('order_id', None)
        if msg_order_id in self.order_map:

            old_order = self.order_map[msg_order_id]
            price = old_order.get('price', None)

            if price in self.price_dict:
                if msg.get('reason', None) == 'canceled':
                    self.price_dict[price].add_cancel(
                        quantity=float(msg.get('remaining_size')), price=price)

                self.price_dict[price].remove_quantity(
                    quantity=old_order['size'], price=price)
                self.price_dict[price].remove_count()

                if self.price_dict[price].count == 0:
                    self.remove_price(price)

            elif RECORD_DATA:
                LOGGER.info('%s remove_order: price not in price_map [%s]' %
                            (msg['product_id'], str(price)))

            del self.order_map[msg_order_id]
