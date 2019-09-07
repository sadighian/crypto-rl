import unittest
import numpy as np
from gym_trading.indicators.indicator import IndicatorManager
from gym_trading.indicators.rsi import RSI
from gym_trading.indicators.tns import TnS


class MyTestCase(unittest.TestCase):

    def test_rsi_up(self):
        indicator = RSI(window=10)
        prices = np.linspace(1, 5, 5)
        indicator.step(price=0)
        for price in prices:
            indicator.step(price=price)
        indicator_value = indicator.get_value()
        self.assertEqual(float(1), indicator_value,
                         msg='indicator_value is {} and should be {}'.format(
                             indicator_value, float(1)))

    def test_rsi_down(self):
        indicator = RSI(window=10)
        prices = np.linspace(1, 5, 5)[::-1]
        indicator.step(price=0)
        for price in prices:
            indicator.step(price=price)
        indicator_value = indicator.get_value()
        self.assertEqual(float(-1), indicator_value,
                         msg='indicator_value is {} and should be {}'.format(
                             indicator_value, float(-1)))

    def test_tns_up(self):
        indicator = TnS(window=10)
        buys = [10]*3 + [0]*7
        sells = [0]*10
        indicator.step(buys=0, sells=0)
        for buy, sell in zip(buys, sells):
            indicator.step(buys=buy, sells=sell)
        indicator_value = indicator.get_value()
        self.assertEqual(float(1), indicator_value,
                         msg='indicator_value is {} and should be {}'.format(
                             indicator_value, float(1)))

    def test_tns_down(self):
        indicator = TnS(window=10)
        buys = [0]*10
        sells = [0]*7 + [10]*3
        indicator.step(buys=0, sells=0)
        for buy, sell in zip(buys, sells):
            indicator.step(buys=buy, sells=sell)
        indicator_value = indicator.get_value()
        self.assertEqual(float(-1), indicator_value,
                         msg='indicator_value is {} and should be {}'.format(
                             indicator_value, float(-1)))

    def test_indicator_manager(self):
        im = IndicatorManager()
        for i in range(2, 5):
            name = 'tns_{}'.format(i)
            print("adding {}".format(name))
            im.add((name, TnS(window=i)))
        buys = [0]*10
        sells = [0]*7 + [10]*3
        im.step(buys=0, sells=0)
        for buy, sell in zip(buys, sells):
            im.step(buys=buy, sells=sell)
        indicator_values = im.get_value()
        self.assertEqual([float(-1)]*3, indicator_values,
                         msg='indicator_value is {} and should be {}'.format(
                             indicator_values, float(-1)))


if __name__ == '__main__':
    unittest.main()
