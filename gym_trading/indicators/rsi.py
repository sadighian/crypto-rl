import numpy as np
from configurations.configs import INDICATOR_WINDOW
from gym_trading.indicators.indicator import Indicator


class RSI(Indicator):

    def __init__(self, window=INDICATOR_WINDOW):
        super(RSI, self).__init__(window=window)
        self.last_price = np.nan
        self.ups = 0.
        self.downs = 0.

    def __str__(self):
        return "***\nRSI: last_price={} | ups={} | downs={}\n***".format(
                   self.last_price, self.ups, self.downs)

    def reset(self):
        self.all_history_queue.clear()
        self.last_price = np.nan
        self.ups = 0.
        self.downs = 0.

    def step(self, price=100.):
        if np.isnan(self.last_price):
            self.last_price = price
            return

        if np.isnan(price):
            print('Error: RSI.step() -> price is {}'.format(price))
            return

        if price == 0.:
            price_pct_change = 0.
        elif self.last_price == 0.:
            price_pct_change = 0.
        else:
            price_pct_change = round((price - self.last_price) / self.last_price, 6)

        if np.isinf(price_pct_change):
            price_pct_change = 0.

        self.last_price = price

        if price_pct_change > 0.:
            self.ups += price_pct_change
        elif price_pct_change < 0.:
            self.downs += price_pct_change

        self.all_history_queue.append(price_pct_change)

        if len(self.all_history_queue) >= self.window:
            price_to_remove = self.all_history_queue.popleft()

            if price_to_remove > 0.:
                self.ups -= price_to_remove
            elif price_to_remove < 0.:
                self.downs -= price_to_remove

    def get_value(self):
        abs_downs = abs(self.downs)
        nom = self.ups - abs_downs
        denom = self.ups + abs_downs
        if denom == 0.:
            return 0.
        elif nom == 0.:
            return 0.
        else:
            return nom / denom
