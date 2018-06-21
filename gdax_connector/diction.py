from datetime import datetime as dt

from sortedcontainers import SortedDict


class Diction(object):

    def __init__(self, sym, side):
        self.price_dict = SortedDict()
        self.order_map = dict()
        self.side = side
        self.sym = sym
        self.max_book_size = 1000

    def __str__(self):
        if self.side == 'asks':
            ask_price, ask_value = self.get_ask()
            message = '%s x %s' % (str(round(ask_price, 2)), str(round(ask_value['size'], 2)))
        else:
            bid_price, bid_value = self.get_bid()
            message = '%s x %s' % (str(round(bid_value['size'], 2)), str(round(bid_price, 2)))
        return message

    def clear(self):
        """
        Reset price tree and order map
        :return: void
        """
        self.price_dict = SortedDict()
        self.order_map = dict()

    def do_next_price(self, side, reference, notional=float(75000)):
        if side == 'asks':
            total_notional = float(0)
            depth = float(0)
            for k, v in self.price_dict.items():
                total_notional += (k * v['size'])
                if total_notional > notional:
                    depth = k
                    break
            return float(depth - reference)
        else:
            total_notional = float(0)
            depth = float(0)
            for k, v in reversed(self.price_dict.items()):
                total_notional += (k * v['size'])
                if total_notional > notional:
                    depth = k
                    break
            return float(reference - depth)

    def create_price(self, price):
        """
        Create new node
        :param price:
        :return:
        """
        self.price_dict[price] = {'size': float(0), 'count': int(0)}

    def remove_price(self, price):
        """
        Remove node
        :param price:
        :return:
        """
        del self.price_dict[price]

    def receive(self, msg):
        """
        add incoming orders to order map
        :param msg:
        :return:
        """
        pass

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

    def insert_orders(self, price, size, oid, product_id, side):
        """
        Used for preloading limit order book
        :param price: order price
        :param size: order size
        :param oid: order id
        :param side: buy or sell
        :param product_id: currency ticker
        :return:
        """
        order = {
            'order_id': oid,
            'price': float(price),
            'size': float(size),
            'side': side,
            'time': dt.now(),
            'type': 'preload',
            'product_id': product_id
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

    def get_ask(self):
        """
        Best offer
        :return: lowest ask (decimal)
        """
        if len(self.price_dict) > 0:
            return self.price_dict.items()[0]
        else:
            return 0.0

    def get_bid(self):
        """
        Best bid
        :return: highest bid (decimal)
        """
        if len(self.price_dict) > 0:
            return self.price_dict.items()[-1]
        else:
            return 0.0

    def get_asks_to_list(self):
        """
        Transform order book to dictionary with 3 lists:
            1- ask prices
            2- cummulative ask volume at a given price
            3- number of orders resting at a given price
        :return: dictionary
        """
        prices, sizes, counts = list(), list(), list()
        counter = 0
        for k, v in self.price_dict.items():
            counter += 1
            if counter > self.max_book_size:
                break
            prices.append(k)
            sizes.append(v['size'])
            counts.append(v['count'])

        # return dict(price=prices, size=sizes, count=counts)
        return prices, sizes, counts

    def get_bids_to_list(self):
        """
        Transform order book to dictionary with 3 lists:
            1- bid prices
            2- cummulative bid volume at a given price
            3- number of orders resting at a given price
        :return: dictionary
        """
        prices, sizes, counts = list(), list(), list()
        counter = 0
        for k, v in reversed(self.price_dict.items()):
            counter += 1
            if counter > self.max_book_size:
                break
            prices.append(k)
            sizes.append(v['size'])
            counts.append(v['count'])

        # return dict(price=prices, size=sizes, count=counts)
        return prices, sizes, counts
