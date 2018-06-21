from sortedcontainers import SortedDict


class Diction(object):

    def __init__(self, sym, side):
        self.price_dict = SortedDict()
        self.order_map = dict()
        self.side = side
        self.sym = sym

    def __str__(self):
        if self.side == 'asks':
            ask_price, ask_value = self.get_ask()
            message = '%.2f x %.2f' % (ask_price, ask_value['size'])
        else:
            bid_price, bid_value = self.get_bid()
            message = '%.2f x %.2f' % (bid_value['size'], bid_price)
        return message

    def clear(self):
        """
        Reset price tree and order map
        :return: void
        """
        self.price_dict = SortedDict()
        self.order_map = dict()

    def do_next_price(self, side, reference, notional=float(75000)):
        total_notional, depth = float(0), float(0)

        if side == 'asks':
            for k, v in self.price_dict.items():
                total_notional += (k * v['size'])
                if total_notional > notional:
                    depth = k
                    break
            return depth - reference

        else:
            for k, v in reversed(self.price_dict.items()):
                total_notional += (k * v['size'])
                if total_notional > notional:
                    depth = k
                    break
            return reference - depth

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

    def insert_order(self, msg):
        """
        Create new node
        :param msg:
        :return:
        """
        self.order_map[msg['order_id']] = msg

        if msg['price'] not in self.price_dict:
            self.create_price(msg['price'])

        self.price_dict[msg['price']]['size'] += abs(msg['size'])
        self.price_dict[msg['price']]['count'] += 1

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

        else:
            print('remove_order: order_id not found %s\n' % msg)

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

        for k, v in self.price_dict.items():
            prices.append(k)
            sizes.append(v['size'])
            counts.append(v['count'])

        return dict(price=prices, size=sizes, count=counts)

    def get_bids_to_list(self):
        """
        Transform order book to dictionary with 3 lists:
            1- bid prices
            2- cummulative bid volume at a given price
            3- number of orders resting at a given price
        :return: dictionary
        """
        prices, sizes, counts = list(), list(), list()

        for k, v in reversed(self.price_dict.items()):
            prices.append(k)
            sizes.append(v['size'])
            counts.append(v['count'])

        return dict(price=prices, size=sizes, count=counts)
