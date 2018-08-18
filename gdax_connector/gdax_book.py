from common_components.book import Book
from datetime import datetime as dt
import pytz as tz


class GdaxBook(Book):

    def __init__(self, sym, side):
        super(GdaxBook, self).__init__(sym, side)

    def insert_order(self, msg):
        """
        Create new node
        :param msg:
        :return:
        """
        if msg['order_id'] not in self.order_map:
            order = {
                'order_id': msg['order_id'],
                'price': float(msg['price']),
                'size': float(msg.get('size') or msg['remaining_size']),
                'side': msg['side'],
                'time': msg['time'],
                'type': msg['type'],
                'product_id': msg['product_id']
            }
            if order['price'] not in self.price_dict:
                self.create_price(order['price'])

            self.price_dict[order['price']]['size'] += order['size']
            self.price_dict[order['price']]['count'] += 1
            self.order_map[order['order_id']] = order

    def match(self, msg):
        """
        Change volume of book
        :param msg:
        :return:
        """
        if msg['maker_order_id'] in self.order_map:
            old_order = self.order_map[msg['maker_order_id']]
            order = {
                'order_id': msg['maker_order_id'],
                'price': float(msg['price']),
                'size': float(msg['size']),
                'side': msg['side'],
                'time': msg['time'],
                'type': msg['type'],
                'product_id': msg['product_id']
            }
            if order['price'] in self.price_dict:
                remove_size = order['size']
                remaining_size = old_order['size'] - remove_size
                order['size'] = remaining_size
                self.order_map[old_order['order_id']] = order
                self.price_dict[old_order['price']]['size'] -= remove_size
            else:
                print('\nmatch: price not in tree already [%s]\n' % msg)
        else:
            print('\n%s match: order id cannot be found for %s\n' % (self.sym, msg))

    def change(self, msg):
        """
        Update inventory
        :param msg:
        :return:
        """
        if 'price' in msg:
            if msg['order_id'] in self.order_map:
                old_order = self.order_map[msg['order_id']]
                new_size = float(msg['new_size'])
                old_size = old_order['size']
                diff = old_size - new_size
                old_order['size'] = new_size
                self.order_map[old_order['order_id']] = old_order
                self.price_dict[old_order['price']]['size'] -= diff
            else:
                print('\n%s change: missing order_ID [%s] from order_map\n' % (self.sym, msg))

    def remove_order(self, msg):
        """
        Done messages result in the order being removed from map
        :param msg:
        :return:
        """
        if msg['order_id'] in self.order_map:
            old_order = self.order_map[msg['order_id']]
            del self.order_map[msg['order_id']]
            if old_order['price'] in self.price_dict:
                self.price_dict[old_order['price']]['size'] -= old_order['size']
                self.price_dict[old_order['price']]['count'] -= 1
                if self.price_dict[old_order['price']]['count'] == 0:
                    self.remove_price(old_order['price'])
            else:
                print('%s remove_order: price not in price_map [%s]' % (msg['product_id'], str(old_order['price'])))

    def get_bids_to_list(self):

        book = dict()
        counter = 0

        for k, v in reversed(self.price_dict.items()):
            if counter >= self.max_book_size:
                book['index'] = dt.now(tz=tz.utc)
                break

            book['gdax_bid_price_%i' % counter] = k
            book['gdax_bid_size_%i' % counter] = v['size']

            counter += 1
        # print(book)
        return book

    def get_asks_to_list(self):

        book = dict()
        counter = 0

        for k, v in self.price_dict.items():
            if counter >= self.max_book_size:
                book['index'] = dt.now(tz=tz.utc)
                break

            book['gdax_ask_price_%i' % counter] = k
            book['gdax_ask_size_%i' % counter] = v['size']

            counter += 1

        return book
