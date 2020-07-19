import unittest

from gym_trading.utils.broker import Broker
from gym_trading.utils.decorator import debugging
from gym_trading.utils.order import LimitOrder, MarketOrder


class MarketOrderTestCases(unittest.TestCase):

    @debugging
    def test_case_one(self):
        print('\nTest_Case_One')

        test_position = Broker()
        midpoint = 100.
        fee = .003

        order_open = MarketOrder(ccy='BTC-USD', side='long', price=midpoint, step=1)
        test_position.add(order=order_open)

        self.assertEqual(1, test_position.long_inventory.position_count)
        print('LONG Unrealized_pnl: %f' % test_position.long_inventory.get_unrealized_pnl(
            price=midpoint))

        self.assertEqual(0, test_position.short_inventory.position_count)
        self.assertEqual(0., test_position.short_inventory.get_unrealized_pnl(
            price=midpoint))

        order_close = MarketOrder(ccy='BTC-USD', side='long',
                                  price=midpoint + (midpoint * fee * 5), step=100)

        test_position.remove(order=order_close)
        self.assertEqual(0, test_position.long_inventory.position_count)
        print('LONG Unrealized_pnl: %f' % test_position.long_inventory.get_unrealized_pnl(
            price=midpoint))

        self.assertEqual(test_position.short_inventory.position_count, 0)
        self.assertEqual(
            test_position.short_inventory.get_unrealized_pnl(price=midpoint), 0.)
        print('LONG Realized_pnl: %f' % test_position.realized_pnl)

    @debugging
    def test_case_two(self):
        print('\nTest_Case_Two')

        test_position = Broker()
        midpoint = 100.
        fee = .003

        order_open = MarketOrder(ccy='BTC-USD', side='short', price=midpoint, step=1)
        test_position.add(order=order_open)
        self.assertEqual(1, test_position.short_inventory.position_count)
        self.assertEqual(0, test_position.long_inventory.position_count)
        self.assertEqual(0., test_position.long_inventory.get_unrealized_pnl(
            price=midpoint))
        print(
            'SHORT Unrealized_pnl: %f' % test_position.short_inventory.get_unrealized_pnl(
                price=midpoint))

        order_close = MarketOrder(ccy='BTC-USD', side='short',
                                  price=midpoint - (midpoint * fee * 15), step=100)
        test_position.remove(order=order_close)
        self.assertEqual(0, test_position.short_inventory.position_count)
        self.assertEqual(0, test_position.long_inventory.position_count)
        self.assertEqual(0., test_position.long_inventory.get_unrealized_pnl(
            price=midpoint))
        print(
            'SHORT Unrealized_pnl: %f' % test_position.short_inventory.get_unrealized_pnl(
                price=midpoint))
        print('SHORT Realized_pnl: %f' % test_position.realized_pnl)

    @debugging
    def test_case_three(self):
        print('\nTest_Case_Three')

        test_position = Broker(5)
        midpoint = 100.

        for i in range(10):
            order_open = MarketOrder(ccy='BTC-USD', side='long', price=midpoint - i, step=i)
            test_position.add(order=order_open)

        self.assertEqual(5, test_position.long_inventory.position_count)
        self.assertEqual(0, test_position.short_inventory.position_count)
        print('Confirm we have 5 positions: %i' % test_position.long_inventory.position_count)

        for i in range(10):
            order_open = MarketOrder(ccy='BTC-USD', side='long', price=midpoint + i, step=i)
            test_position.remove(order=order_open)

        self.assertEqual(0, test_position.long_inventory.position_count)
        self.assertEqual(0, test_position.short_inventory.position_count)


class LimitOrderTestCases(unittest.TestCase):

    @debugging
    def test_long_pnl(self):
        test_position = Broker()
        step = 0
        bid_price = 101.
        ask_price = 102.
        buy_volume = 100
        sell_volume = 100
        pnl = 0.

        def walk_forward(pnl, step, bid_price, ask_price, buy_volume, sell_volume, down=True):
            for i in range(50):
                step += 1
                if down:
                    bid_price *= 0.99
                    ask_price *= 0.99
                else:
                    bid_price *= 1.01
                    ask_price *= 1.01

                pnl, is_long_order_filled, is_short_order_filled = \
                    test_position.step_limit_order_pnl(
                        bid_price=bid_price, ask_price=ask_price, buy_volume=buy_volume,
                        sell_volume=sell_volume, step=step)
                pnl += pnl
                if i % 10 == 0:
                    print('bid_price={:.2f} | ask_price={:.2f}'.format(bid_price,
                                                                       ask_price))
            return step, bid_price, ask_price, buy_volume, sell_volume, pnl

        test_position.add(
            order=LimitOrder(ccy='BTC-USD', side='long', price=100., step=step,
                             queue_ahead=1000))

        step, _, _, buy_volume, sell_volume, pnl = walk_forward(pnl, step, bid_price,
                                                                ask_price, buy_volume,
                                                                sell_volume, down=True)
        self.assertEqual(1, test_position.long_inventory_count)

        test_position.add(
            order=LimitOrder(ccy='BTC-USD', side='short', price=105., step=step,
                             queue_ahead=0))
        _, _, _, _, _, pnl = walk_forward(pnl, step, bid_price, ask_price, buy_volume,
                                          sell_volume, down=False)
        realized_pnl = round(test_position.realized_pnl, 3)

        self.assertEqual(0.05, realized_pnl,
                         "Expected Realized PnL of 0.5 and got {}".format(realized_pnl))
        self.assertEqual(0,
                         test_position.short_inventory_count +
                         test_position.long_inventory_count)
        print("PnL: {}".format(pnl))

    @debugging
    def test_avg_exe(self):
        test_position = Broker()

        # perform a partial fill on the first order
        step = 0
        bid_price = 101.
        ask_price = 102.
        buy_volume = 500
        sell_volume = 500

        test_position.add(
            order=LimitOrder(ccy='BTC-USD', side='long', price=bid_price, step=step,
                             queue_ahead=0))

        print("taking first step...")
        step += 1
        pnl, is_long_order_filled, is_short_order_filled = \
            test_position.step_limit_order_pnl(
                bid_price=bid_price, ask_price=ask_price, buy_volume=buy_volume,
                sell_volume=sell_volume, step=step)
        pnl += pnl
        self.assertEqual(500, test_position.long_inventory.order.executed)
        self.assertEqual(0, test_position.long_inventory_count)

        # if order gets filled with a bid below the order's price, the order should NOT
        # receive any price improvement during the execution.
        bid_price = 99.
        ask_price = 100.
        test_position.add(
            order=LimitOrder(ccy='BTC-USD', side='long', price=bid_price, step=step,
                             queue_ahead=0))

        print("taking second step...")
        step += 1
        pnl, is_long_order_filled, is_short_order_filled = \
            test_position.step_limit_order_pnl(
                bid_price=bid_price, ask_price=ask_price, buy_volume=buy_volume,
                sell_volume=sell_volume, step=step)
        pnl += pnl
        self.assertEqual(1, test_position.long_inventory_count)
        self.assertEqual(100., test_position.long_inventory.average_price)
        print("PnL: {}".format(pnl))

    @debugging
    def test_lob_queuing(self):
        test_position = Broker()

        # perform a partial fill on the first order
        step = 0
        bid_price = 102.
        ask_price = 103.
        buy_volume = 500
        sell_volume = 500
        queue_ahead = 800

        order_open = LimitOrder(ccy='BTC-USD', side='long', price=bid_price, step=step,
                                queue_ahead=queue_ahead)
        test_position.add(order=order_open)

        step += 1
        pnl, is_long_order_filled, is_short_order_filled = \
            test_position.step_limit_order_pnl(
                bid_price=bid_price, ask_price=ask_price, buy_volume=buy_volume,
                sell_volume=sell_volume, step=step)
        pnl += pnl

        print("#1 long_inventory.order = \n{}".format(test_position.long_inventory.order))
        self.assertEqual(300, test_position.long_inventory.order.queue_ahead)
        self.assertEqual(0, test_position.long_inventory.order.executed)
        self.assertEqual(0, test_position.long_inventory_count)

        step += 1
        pnl, is_long_order_filled, is_short_order_filled = \
            test_position.step_limit_order_pnl(
                bid_price=bid_price, ask_price=ask_price, buy_volume=buy_volume,
                sell_volume=sell_volume, step=step)
        pnl += pnl

        print("#2 long_inventory.order = \n{}".format(test_position.long_inventory.order))
        self.assertEqual(200, test_position.long_inventory.order.executed)
        self.assertEqual(0, test_position.long_inventory_count)

        # if order gets filled with a bid below the order's price, the order should NOT
        # receive any price improvement during the execution.
        bid_price = 100.
        ask_price = 102.
        order_open = LimitOrder(ccy='BTC-USD', side='long', price=bid_price, step=step,
                                queue_ahead=queue_ahead)
        test_position.add(order=order_open)
        print("#3 long_inventory.order = \n{}".format(test_position.long_inventory.order))
        self.assertEqual(0, test_position.long_inventory_count)

        bid_price = 100.
        for i in range(5):
            step += 1
            pnl, is_long_order_filled, is_short_order_filled = \
                test_position.step_limit_order_pnl(
                    bid_price=bid_price, ask_price=ask_price, buy_volume=buy_volume,
                    sell_volume=sell_volume, step=step)
            pnl += pnl

        self.assertEqual(1, test_position.long_inventory_count)
        self.assertEqual(100.40, round(test_position.long_inventory.average_price, 2))
        print("PnL: {}".format(pnl))

    @debugging
    def test_queues_ahead_features(self):
        test_position = Broker()

        # perform a partial fill on the first order
        step = 0
        bid_price = 100.
        ask_price = 200.
        buy_volume = 0
        sell_volume = 0

        order_open_long = LimitOrder(ccy='BTC-USD', side='long', price=bid_price,
                                     step=step, queue_ahead=0)
        order_open_short = LimitOrder(ccy='BTC-USD', side='short', price=ask_price,
                                      step=step, queue_ahead=2000)
        print('opening long position = {}'.format(order_open_long))
        test_position.add(order=order_open_long)
        print('opening short position = {}'.format(order_open_short))
        test_position.add(order=order_open_short)

        print('\ntaking first step...')
        step += 1
        pnl, is_long_order_filled, is_short_order_filled = \
            test_position.step_limit_order_pnl(
                bid_price=bid_price, ask_price=ask_price, buy_volume=buy_volume,
                sell_volume=sell_volume, step=step)
        pnl += pnl

        print("#1 long_inventory.order = \n{}".format(test_position.long_inventory.order))
        print(
            "#1 short_inventory.order = \n{}".format(test_position.short_inventory.order))
        bid_queue, ask_queue = test_position.get_queues_ahead_features()
        print("#1 get_queues_ahead_features:\nbid_queue={} || ask_queue={}".format(
            bid_queue, ask_queue))
        self.assertEqual(0., bid_queue)
        self.assertEqual(-0.67, round(ask_queue, 2))

        print('\ntaking second step...')
        buy_volume = 500
        sell_volume = 500
        step += 1
        pnl, is_long_order_filled, is_short_order_filled = \
            test_position.step_limit_order_pnl(
                bid_price=bid_price, ask_price=ask_price, buy_volume=buy_volume,
                sell_volume=sell_volume, step=step)
        pnl += pnl

        print("#2 long_inventory.order = \n{}".format(test_position.long_inventory.order))
        print(
            "#2 short_inventory.order = \n{}".format(test_position.short_inventory.order))
        bid_queue, ask_queue = test_position.get_queues_ahead_features()
        print("#2 get_queues_ahead_features:\nbid_queue={} || ask_queue={}".format(
            bid_queue, ask_queue))
        self.assertEqual(0.5, bid_queue)
        self.assertEqual(-0.6, round(ask_queue, 2))

        print('\ntaking third step...')
        buy_volume = 500
        sell_volume = 499
        step += 1
        pnl, is_long_order_filled, is_short_order_filled = \
            test_position.step_limit_order_pnl(
                bid_price=bid_price, ask_price=ask_price, buy_volume=buy_volume,
                sell_volume=sell_volume, step=step)
        pnl += pnl

        print("#3 long_inventory.order = \n{}".format(test_position.long_inventory.order))
        print(
            "#3 short_inventory.order = \n{}".format(test_position.short_inventory.order))
        bid_queue, ask_queue = test_position.get_queues_ahead_features()
        print("#3 get_queues_ahead_features:\nbid_queue={} || ask_queue={}".format(
            bid_queue, ask_queue))
        self.assertEqual(0.999, bid_queue)
        self.assertEqual(-0.5, round(ask_queue, 2))

        print('\ntaking fourth step...')
        buy_volume = 500
        sell_volume = 500
        step += 1
        pnl, is_long_order_filled, is_short_order_filled = \
            test_position.step_limit_order_pnl(
                bid_price=bid_price, ask_price=ask_price, buy_volume=buy_volume,
                sell_volume=sell_volume, step=step)
        pnl += pnl

        print("#4 long_inventory.order = \n{}".format(test_position.long_inventory.order))
        print(
            "#4 short_inventory.order = \n{}".format(test_position.short_inventory.order))
        bid_queue, ask_queue = test_position.get_queues_ahead_features()
        print("#4 get_queues_ahead_features:\nbid_queue={} || ask_queue={}".format(
            bid_queue, ask_queue))
        self.assertEqual(0.0, bid_queue)
        self.assertEqual(-0.33, round(ask_queue, 2))
        print("PnL: {}".format(pnl))


if __name__ == '__main__':
    unittest.main()
