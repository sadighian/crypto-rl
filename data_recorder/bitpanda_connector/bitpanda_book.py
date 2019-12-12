from data_recorder.connector_components .book import Book, round_price
from configurations.configs import RECORD_DATA


class CoinbaseBook(Book):

    def __init__(self, sym, side):
        super(CoinbaseBook, self).__init__(sym, side)

    def insert_order(self, msg):
        """
        Create new node
        :param msg:
        :return:
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
            adj_price = round_price(price)

            if adj_price not in self.price_dict:
                self.create_price(adj_price)

            self.price_dict[adj_price].add_limit(quantity=size,
                                                 price=price)
            self.price_dict[adj_price].add_quantity(quantity=size,
                                                    price=price)
            self.price_dict[adj_price].add_count()

    def match(self, msg):
        """
        Change volume of book
        :param msg:
        :return:
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
            adj_price = round_price(order['price'])
            if adj_price in self.price_dict:
                remove_size = order['size']
                remaining_size = old_order['size'] - remove_size
                order['size'] = remaining_size
                self.order_map[old_order['order_id']] = order
                old_order_price = old_order.get('price', None)
                self.price_dict[adj_price].add_market(quantity=remove_size,
                                                      price=old_order_price)
                self.price_dict[adj_price].remove_quantity(quantity=remove_size,
                                                           price=old_order_price)
            else:
                print('\nmatch: price not in tree already [%s]\n' % msg)
        elif RECORD_DATA:
            print('\n%s match: order id cannot be found for %s\n' % (self.sym, msg))

    def change(self, msg):
        """
        Update inventory
        :param msg:
        :return:
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
                adj_price = round_price(old_order_price)
                self.price_dict[adj_price].remove_quantity(quantity=diff,
                                                           price=old_order_price)
            elif RECORD_DATA:
                print('\n%s change: missing order_ID [%s] from order_map\n' %
                      (self.sym, msg))

    def remove_order(self, msg):
        """
        Done messages result in the order being removed from map
        :param msg:
        :return:
        """
        msg_order_id = msg.get('order_id', None)
        if msg_order_id in self.order_map:

            old_order = self.order_map[msg_order_id]
            price = old_order.get('price', None)
            adj_price = round_price(price)

            if adj_price in self.price_dict:
                if msg.get('reason', None) == 'canceled':
                    self.price_dict[adj_price].add_cancel(
                        quantity=float(msg.get('remaining_size')), price=price)

                self.price_dict[adj_price].remove_quantity(
                    quantity=old_order['size'], price=price)
                self.price_dict[adj_price].remove_count()

                if self.price_dict[adj_price].count == 0:
                    self.remove_price(adj_price)

            elif RECORD_DATA:
                print('%s remove_order: price not in price_map [%s]' %
                      (msg['product_id'], str(adj_price)))

            del self.order_map[msg_order_id]
