from indicators.indicator import Indicator


class TnS(Indicator):

    def __init__(self, **kwargs):
        super(TnS, self).__init__(**kwargs)
        self.ups = self.downs = 0.

    def __str__(self):
        return "TNS: ups={} | downs={}".format(self.ups, self.downs)

    def reset(self):
        self.ups = self.downs = 0.
        super(TnS, self).reset()

    def step(self, buys=0., sells=0.):
        self.ups += buys
        self.downs += sells
        self.all_history_queue.append((buys, sells))

        if len(self.all_history_queue) >= self.window:
            buys_, sells_ = self.all_history_queue.popleft()
            self.ups -= buys_
            self.downs -= sells_

        # Save current time step value for EMA, in case smoothing is enabled
        self._value = self.calculate()
        super(TnS, self).step(value=self._value)

    def calculate(self) -> float:
        nom = round(self.ups - self.downs, 6)
        denom = round(self.ups + self.downs, 6)
        return self._divide(nom=nom, denom=denom)
