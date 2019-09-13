import unittest
from gym_trading.utils.mm_broker import Broker, Order


class MarketMakerBrokerTestCases(unittest.TestCase):

    def test_long_pnl(self):
        test_position = Broker()
        step = 0
        bid_price = 101.
        ask_price = 102.
        buy_volume = 100
        sell_volume = 100

        def walk_forward(step, bid_price, ask_price, buy_volume, sell_volume, down=True):
            for i in range(50):
                step += 1
                if down:
                    bid_price *= 0.99
                    ask_price *= 0.99
                else:
                    bid_price *= 1.01
                    ask_price *= 1.01

                test_position.step(bid_price=bid_price, ask_price=ask_price,
                                   buy_volume=buy_volume, sell_volume=sell_volume,
                                   step=step)
                if i % 10 == 0:
                    print('bid_price={:.2f} | ask_price={:.2f}'.format(
                        bid_price, ask_price))
            return step, bid_price, ask_price, buy_volume, sell_volume

        order_open = Order(ccy='BTC-USD', side='long', price=100.,
                           step=step, queue_ahead=1000)
        test_position.add(order=order_open)

        step, _, _, buy_volume, sell_volume = walk_forward(step,
                                                           bid_price,
                                                           ask_price,
                                                           buy_volume,
                                                           sell_volume,
                                                           down=True)
        self.assertEqual(1, test_position.long_inventory_count)

        order_open = Order(ccy='BTC-USD', side='short', price=105., step=step,
                           queue_ahead=0)
        test_position.add(order=order_open)
        _, _, _, _, _ = walk_forward(step,
                                     bid_price,
                                     ask_price,
                                     buy_volume,
                                     sell_volume,
                                     down=False)
        realized_pnl = test_position.get_realized_pnl()
        self.assertEqual(0.048, realized_pnl,
                         "Expected Realized PnL of 0.5 and got {}".format(realized_pnl))
        self.assertEqual(0,
                         test_position.short_inventory_count +
                         test_position.long_inventory_count)

    def test_avg_exe(self):
        test_position = Broker()

        # perform a partial fill on the first order
        step = 0
        bid_price = 101.
        ask_price = 102.
        buy_volume = 500
        sell_volume = 500

        order_open = Order(ccy='BTC-USD', side='long', price=bid_price,
                           step=step, queue_ahead=0)
        test_position.add(order=order_open)

        print("taking first step...")
        step += 1
        test_position.step(bid_price=bid_price, ask_price=ask_price,
                           buy_volume=buy_volume, sell_volume=sell_volume, step=step)
        self.assertEqual(500, test_position.long_inventory.order.executed)
        self.assertEqual(0, test_position.long_inventory_count)

        # if order gets filled with a bid below the order's price, the order should NOT
        # receive any price improvement during the execution.
        bid_price = 100.
        ask_price = 102.

        print("taking second step...")
        step += 1
        test_position.step(bid_price=bid_price, ask_price=ask_price,
                           buy_volume=buy_volume, sell_volume=sell_volume, step=step)
        self.assertEqual(1, test_position.long_inventory_count)
        self.assertEqual(101,
                         test_position.long_inventory.average_price)

    def test_lob_queuing(self):
        test_position = Broker()

        # perform a partial fill on the first order
        step = 0
        bid_price = 102.
        ask_price = 103.
        buy_volume = 500
        sell_volume = 500
        queue_ahead = 800

        order_open = Order(ccy='BTC-USD', side='long', price=bid_price,
                           step=step, queue_ahead=queue_ahead)
        test_position.add(order=order_open)

        step += 1
        test_position.step(bid_price=bid_price, ask_price=ask_price,
                           buy_volume=buy_volume, sell_volume=sell_volume, step=step)

        print("#1 long_inventory.order = \n{}".format(test_position.long_inventory.order))
        self.assertEqual(300, test_position.long_inventory.order.queue_ahead)
        self.assertEqual(0, test_position.long_inventory.order.executed)
        self.assertEqual(0, test_position.long_inventory_count)

        step += 1
        test_position.step(bid_price=bid_price, ask_price=ask_price,
                           buy_volume=buy_volume, sell_volume=sell_volume, step=step)

        print("#2 long_inventory.order = \n{}".format(test_position.long_inventory.order))
        self.assertEqual(200, test_position.long_inventory.order.executed)
        self.assertEqual(0, test_position.long_inventory_count)

        # if order gets filled with a bid below the order's price, the order should NOT
        # receive any price improvement during the execution.
        bid_price = 100.
        ask_price = 102.
        order_open = Order(ccy='BTC-USD', side='long', price=bid_price, step=step,
                           queue_ahead=queue_ahead)
        test_position.add(order=order_open)
        print("#3 long_inventory.order = \n{}".format(test_position.long_inventory.order))
        self.assertEqual(0, test_position.long_inventory_count)

        bid_price = 100.
        for i in range(5):
            step += 1
            test_position.step(bid_price=bid_price, ask_price=ask_price,
                               buy_volume=buy_volume, sell_volume=sell_volume, step=step)

        self.assertEqual(1, test_position.long_inventory_count)
        self.assertEqual(100.39,
                         round(test_position.long_inventory.average_price, 2))

    def test_queues_ahead_features(self):
        test_position = Broker()

        # perform a partial fill on the first order
        step = 0
        bid_price = 100.
        ask_price = 200.
        buy_volume = 0
        sell_volume = 0

        order_open_long = Order(ccy='BTC-USD', side='long', price=bid_price,
                           step=step, queue_ahead=0)
        order_open_short = Order(ccy='BTC-USD', side='short', price=ask_price,
                                 step=step, queue_ahead=2000)
        print('opening long position')
        test_position.add(order=order_open_long)
        print('opening short position')
        test_position.add(order=order_open_short)

        print('\ntaking first step...')
        step += 1
        test_position.step(bid_price=bid_price, ask_price=ask_price,
                           buy_volume=buy_volume, sell_volume=sell_volume, step=step)

        print("#1 long_inventory.order = \n{}".format(
            test_position.long_inventory.order))
        print("#1 short_inventory.order = \n{}".format(
            test_position.short_inventory.order))
        bid_queue, ask_queue = test_position.get_queues_ahead_features()
        print("#1 get_queues_ahead_features:\nbid_queue={} || ask_queue={}".format(
            bid_queue, ask_queue))
        self.assertEqual(0., bid_queue)
        self.assertEqual(-0.67, round(ask_queue,2))

        print('\ntaking second step...')
        buy_volume = 500
        sell_volume = 500
        step += 1
        test_position.step(bid_price=bid_price, ask_price=ask_price,
                           buy_volume=buy_volume, sell_volume=sell_volume, step=step)

        print("#2 long_inventory.order = \n{}".format(
            test_position.long_inventory.order))
        print("#2 short_inventory.order = \n{}".format(
                test_position.short_inventory.order))
        bid_queue, ask_queue = test_position.get_queues_ahead_features()
        print("#2 get_queues_ahead_features:\nbid_queue={} || ask_queue={}".format(
            bid_queue, ask_queue))
        self.assertEqual(0.5, bid_queue)
        self.assertEqual(-0.6, round(ask_queue, 2))

        print('\ntaking third step...')
        buy_volume = 500
        sell_volume = 499
        step += 1
        test_position.step(bid_price=bid_price, ask_price=ask_price,
                           buy_volume=buy_volume, sell_volume=sell_volume, step=step)

        print("#3 long_inventory.order = \n{}".format(
            test_position.long_inventory.order))
        print("#3 short_inventory.order = \n{}".format(
                test_position.short_inventory.order))
        bid_queue, ask_queue = test_position.get_queues_ahead_features()
        print("#3 get_queues_ahead_features:\nbid_queue={} || ask_queue={}".format(
            bid_queue, ask_queue))
        self.assertEqual(0.999, bid_queue)
        self.assertEqual(-0.5, round(ask_queue, 2))

        print('\ntaking fourth step...')
        buy_volume = 500
        sell_volume = 500
        step += 1
        test_position.step(bid_price=bid_price, ask_price=ask_price,
                           buy_volume=buy_volume, sell_volume=sell_volume, step=step)

        print("#4 long_inventory.order = \n{}".format(
            test_position.long_inventory.order))
        print("#4 short_inventory.order = \n{}".format(
                test_position.short_inventory.order))
        bid_queue, ask_queue = test_position.get_queues_ahead_features()
        print("#4 get_queues_ahead_features:\nbid_queue={} || ask_queue={}".format(
            bid_queue, ask_queue))
        self.assertEqual(0.0, bid_queue)
        self.assertEqual(-0.33, round(ask_queue, 2))

        print("\ndone.")


if __name__ == '__main__':
    unittest.main()
