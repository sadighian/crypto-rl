from configurations.configs import INDICATOR_WINDOW
from indicators.indicator import Indicator
from collections import deque
import numpy as np


class PnlNorm(Indicator):

    def __init__(self, window=INDICATOR_WINDOW, alpha=None):
        super(PnlNorm, self).__init__(window=window, alpha=alpha)
        self.last_pnl = np.nan
        self.ups = deque(maxlen=self.window)
        self.downs = deque(maxlen=self.window)

    def __str__(self):
        return "***\nRSI: last_pnl={} | ups={} | downs={}\n***".format(
                   self.last_pnl, self.ups, self.downs)

    def reset(self):
        self.last_pnl = np.nan
        self.ups.clear()
        self.downs.clear()
        super(PnlNorm, self).reset()

    def step(self, pnl: float):
        # handle NaNs
        if np.isnan(self.last_pnl):
            self.last_pnl = pnl
            return

        if np.isnan(pnl):
            print('Error: RSI.step() -> price is {}'.format(pnl))
            return

        if pnl == 0.:
            pnl_pct_change = 0.
        elif self.last_pnl == 0.:
            pnl_pct_change = 0.
        else:
            pnl_pct_change = round((pnl / self.last_pnl) - 1., 6)

        if np.isinf(pnl_pct_change):
            pnl_pct_change = 0.

        # save mid-price change
        self.last_pnl = pnl

        # add indicators to queue
        self.ups.append(max(0., pnl_pct_change))
        self.downs.append(abs(min(0., pnl_pct_change)))

        if len(self.ups) >= self.window:
            _ = self.ups.popleft()
            _ = self.downs.popleft()

        # Save current time step value for EMA, in case smoothing is enabled
        self._value = self.calculate()
        super(PnlNorm, self).step(value=self._value)

    @property
    def _mean_up(self):
        """
        Note: calculated in O(n) time complexity, not O(1)
        :return:
        """
        return np.mean(self.ups)

    @property
    def _mean_down(self):
        """
        Note: calculated in O(n) time complexity, not O(1)
        :return:
        """
        return np.mean(self.downs)

    @property
    def _std_up(self):
        """
        Note: calculated in O(n) time complexity, not O(1)
        :return:
        """
        return np.std(self.ups)

    @property
    def _std_down(self):
        """
        Note: calculated in O(n) time complexity, not O(1)
        :return:
        """
        return np.std(self.downs)

    def calculate(self):
        sd = self._std_down
        su = self._std_up
        nom = self._mean_up * sd - self._mean_down * su
        denom = su + sd
        return self._divide(nom=nom, denom=denom)

