import unittest
from gym_trading.utils.broker import Broker, Order


class PriceJumpBrokerTestCases(unittest.TestCase):

    def test_case_one(self):
        print('\nTest_Case_One')

        test_position = Broker()
        midpoint = 100.
        fee = .003

        order_open = Order(ccy='BTC-USD', side='long', price=midpoint, step=1)
        test_position.add(order=order_open)

        self.assertEqual(1, test_position.long_inventory.position_count)
        print('LONG Unrealized_pnl: %f' %
              test_position.long_inventory.get_unrealized_pnl())

        self.assertEqual(0, test_position.short_inventory.position_count)
        self.assertEqual(0., test_position.short_inventory.get_unrealized_pnl())

        order_close = Order(ccy='BTC-USD', side='long', price=midpoint + (
                midpoint*fee*5), step=100)

        test_position.remove(order=order_close)
        self.assertEqual(0, test_position.long_inventory.position_count)
        print('LONG Unrealized_pnl: %f' %
              test_position.long_inventory.get_unrealized_pnl())

        self.assertEqual(test_position.short_inventory.position_count, 0)
        self.assertEqual(test_position.short_inventory.get_unrealized_pnl(), 0.)
        print('LONG Realized_pnl: %f' % test_position.get_realized_pnl())

    def test_case_two(self):
        print('\nTest_Case_Two')

        test_position = Broker()
        midpoint = 100.
        fee = .003

        order_open = Order(ccy='BTC-USD', side='short', price=midpoint, step=1)
        test_position.add(order=order_open)
        self.assertEqual(1, test_position.short_inventory.position_count)
        self.assertEqual(0, test_position.long_inventory.position_count)
        self.assertEqual(0., test_position.long_inventory.get_unrealized_pnl())
        print('SHORT Unrealized_pnl: %f' %
              test_position.short_inventory.get_unrealized_pnl())

        order_close = Order(ccy='BTC-USD', side='short',
                            price=midpoint - (midpoint*fee*15), step=100)
        test_position.remove(order=order_close)
        self.assertEqual(0, test_position.short_inventory.position_count)
        self.assertEqual(0, test_position.long_inventory.position_count)
        self.assertEqual(0., test_position.long_inventory.get_unrealized_pnl())
        print('SHORT Unrealized_pnl: %f' %
              test_position.short_inventory.get_unrealized_pnl())
        print('SHORT Realized_pnl: %f' % test_position.get_realized_pnl())

    def test_case_three(self):
        print('\nTest_Case_Three')

        test_position = Broker(5)
        midpoint = 100.

        for i in range(10):
            order_open = Order(ccy='BTC-USD', side='long', price=midpoint - i, step=i)
            test_position.add(order=order_open)

        self.assertEqual(5, test_position.long_inventory.position_count)
        self.assertEqual(0, test_position.short_inventory.position_count)
        print('Confirm we have 5 positions: %i' %
              test_position.long_inventory.position_count)

        for i in range(10):
            order_open = Order(ccy='BTC-USD', side='long', price=midpoint + i, step=i)
            test_position.remove(order=order_open)

        self.assertEqual(0, test_position.long_inventory.position_count)
        self.assertEqual(0, test_position.short_inventory.position_count)


if __name__ == '__main__':
    unittest.main()
