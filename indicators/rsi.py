import numpy as np

from indicators.indicator import Indicator


class RSI(Indicator):
    """
    Price change momentum indicator. Note: Scaled to [-1, 1] and not [0, 100].
    """

    def __init__(self, **kwargs):
        super().__init__(label='rsi', **kwargs)
        self.last_price = None
        self.ups = self.downs = 0.

    def __str__(self):
        return f"RSI: [ last_price = {self.last_price} | " \
               f"ups = {self.ups} | downs = {self.downs} ]"

    def reset(self) -> None:
        """
        Reset the indicator.

        :return:
        """
        self.last_price = None
        self.ups = self.downs = 0.
        super().reset()

    def step(self, price: float) -> None:
        """
        Update indicator value incrementally.

        :param price: midpoint price
        :return:
        """
        if self.last_price is None:
            self.last_price = price
            return

        if np.isnan(price):
            print(f'Error: RSI.step() -> price is {price}')
            return

        if price == 0.:
            price_pct_change = 0.
        elif self.last_price == 0.:
            price_pct_change = 0.
        else:
            price_pct_change = round((price / self.last_price) - 1., 6)

        if np.isinf(price_pct_change):
            price_pct_change = 0.

        self.last_price = price

        if price_pct_change > 0.:
            self.ups += price_pct_change
        else:
            self.downs += price_pct_change

        self.all_history_queue.append(price_pct_change)

        # only pop off items if queue is done warming up
        if len(self.all_history_queue) <= self.window:
            return

        price_to_remove = self.all_history_queue.popleft()

        if price_to_remove > 0.:
            self.ups -= price_to_remove
        else:
            self.downs -= price_to_remove

        # Save current time step value for EMA, in case smoothing is enabled
        self._value = self.calculate()
        super().step(value=self._value)

    def calculate(self) -> float:
        """
        Calculate price momentum imbalance.

        :return: imbalance in range of [-1, 1]
        """
        mean_downs = abs(self.safe_divide(nom=self.downs, denom=self.window))
        mean_ups = self.safe_divide(nom=self.ups, denom=self.window)
        gain = mean_ups - mean_downs
        loss = mean_ups + mean_downs
        return self.safe_divide(nom=gain, denom=loss)
