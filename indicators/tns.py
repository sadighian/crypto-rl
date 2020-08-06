from indicators.indicator import Indicator


class TnS(Indicator):
    """
    Time and sales [trade flow] imbalance indicator
    """

    def __init__(self, **kwargs):
        super().__init__(label='tns', **kwargs)
        self.ups = self.downs = 0.

    def __str__(self):
        return f"TNS: ups={self.ups} | downs={self.downs}"

    def reset(self) -> None:
        """
        Reset indicator.
        """
        self.ups = self.downs = 0.
        super().reset()

    def step(self, buys: float, sells: float) -> None:
        """
        Update indicator with new transaction data.

        :param buys: buy transactions
        :param sells: sell transactions
        """
        self.ups += abs(buys)
        self.downs += abs(sells)
        self.all_history_queue.append((buys, sells))

        # only pop off items if queue is done warming up
        if len(self.all_history_queue) <= self.window:
            return

        buys_, sells_ = self.all_history_queue.popleft()
        self.ups -= abs(buys_)
        self.downs -= abs(sells_)

        # Save current time step value for EMA, in case smoothing is enabled
        self._value = self.calculate()
        super().step(value=self._value)

    def calculate(self) -> float:
        """
        Calculate trade flow imbalance.

        :return: imbalance in range of [-1, 1]
        """
        gain = round(self.ups - self.downs, 6)
        loss = round(self.ups + self.downs, 6)
        return self.safe_divide(nom=gain, denom=loss)
