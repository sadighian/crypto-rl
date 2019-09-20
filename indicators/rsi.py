import numpy as np
from configurations.configs import INDICATOR_WINDOW
from indicators.indicator import Indicator


class RSI(Indicator):

    def __init__(self, window=INDICATOR_WINDOW, alpha=None):
        super(RSI, self).__init__(window=window, alpha=alpha)
        self.last_price = np.nan
        self.ups = self.downs = 0.

    def __str__(self):
        return "***\nRSI: last_price={} | ups={} | downs={}\n***".format(
                   self.last_price, self.ups, self.downs)

    def reset(self):
        self.last_price = np.nan
        self.ups = self.downs = 0.
        super(RSI, self).reset()

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

        # Save current time step value for EMA, in case smoothing is enabled
        self._value = self.calculate()
        super(RSI, self).step(value=self._value)

    def calculate(self):
        abs_downs = abs(self.downs)
        nom = self.ups - abs_downs
        denom = self.ups + abs_downs
        return self._divide(nom=nom, denom=denom)
