import unittest

import numpy as np

from gym_trading.utils.decorator import print_time
from indicators.indicator import IndicatorManager
from indicators.rsi import RSI
from indicators.tns import TnS


class IndicatorTestCases(unittest.TestCase):

    @print_time
    def test_rsi_up(self):
        indicator = RSI(window=10)
        prices = np.linspace(1, 5, 50)
        indicator.step(price=0)
        for price in prices:
            indicator.step(price=price)
        indicator_value = indicator.value
        self.assertEqual(float(1), indicator_value,
                         msg='indicator_value is {} and should be {}'.format(
                             indicator_value, float(1)))

    @print_time
    def test_rsi_down(self):
        indicator = RSI(window=10)
        prices = np.linspace(1, 5, 50)[::-1]
        indicator.step(price=0)
        for price in prices:
            indicator.step(price=price)
        indicator_value = indicator.value
        self.assertEqual(float(-1), indicator_value,
                         msg='indicator_value is {} and should be {}'.format(
                             indicator_value, float(-1)))

    @print_time
    def test_tns_up(self):
        indicator = TnS(window=10)
        buys = [10] * 3 + [0] * 7
        sells = [0] * 10
        indicator.step(buys=0, sells=0)
        for buy, sell in zip(buys, sells):
            indicator.step(buys=buy, sells=sell)
        indicator_value = indicator.value
        self.assertEqual(float(1), indicator_value,
                         msg='indicator_value is {} and should be {}'.format(
                             indicator_value, float(1)))

    @print_time
    def test_tns_down(self):
        indicator = TnS(window=10)
        buys = [0] * 10
        sells = [0] * 7 + [10] * 3
        indicator.step(buys=0, sells=0)
        for buy, sell in zip(buys, sells):
            indicator.step(buys=buy, sells=sell)
        indicator_value = indicator.value
        self.assertEqual(float(-1), indicator_value,
                         msg='indicator_value is {} and should be {}'.format(
                             indicator_value, float(-1)))

    @print_time
    def test_indicator_manager(self):
        im = IndicatorManager()
        for i in range(2, 5):
            name = 'tns_{}'.format(i)
            print("adding {}".format(name))
            im.add((name, TnS(window=i)))

        buys = [0] * 10
        sells = [0] * 7 + [10] * 3
        im.step(buys=0, sells=0)
        for buy, sell in zip(buys, sells):
            im.step(buys=buy, sells=sell)
        indicator_values = im.get_value()
        self.assertEqual([float(-1)] * 3, indicator_values,
                         msg='indicator_value is {} and should be {}'.format(
                             indicator_values, float(-1)))

    @print_time
    def test_exponential_moving_average(self):
        indicator_ema = RSI(window=10, alpha=0.99)
        indicator = RSI(window=10, alpha=None)
        prices = np.concatenate((np.linspace(1, 5, 20)[::-1], np.linspace(1, 5, 20)),
                                axis=0)

        indicator.step(price=0)
        for price in prices:
            indicator.step(price=price)
            indicator_ema.step(price=price)
        indicator_value = indicator.value
        indicator_ema_value = indicator_ema.value
        print("indicator_value: {:.6f} | ema: {:.6f}".format(indicator_value,
                                                             indicator_ema_value))

        self.assertNotAlmostEqual(indicator_value, indicator_ema_value,
                                  msg='indicator_value is {} and should be {}'.format(
                                      indicator_value, indicator_ema_value))

        self.assertNotAlmostEqual(1., indicator_ema_value,
                                  msg='indicator_ema_value is {} and should be {}'.format(
                                      indicator_ema_value, 1.))

        self.assertAlmostEqual(1., indicator_value,
                               msg='indicator_value is {} and should be {}'.format(
                                   indicator_value, 1.))


if __name__ == '__main__':
    unittest.main()
