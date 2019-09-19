from configurations.configs import INDICATOR_WINDOW
from indicators.indicator import Indicator


class TnS(Indicator):

    def __init__(self, window=INDICATOR_WINDOW, alpha=None):
        super(TnS, self).__init__(window=window, alpha=alpha)
        self.ups = 0.
        self.downs = 0.

    def __str__(self):
        return "TNS: ups={} | downs={}".format(self.ups, self.downs)

    def reset(self):
        self.all_history_queue.clear()
        self.ups = 0.
        self.downs = 0.

    def step(self, buys=0., sells=0.):
        self.ups += buys
        self.downs += sells
        self.all_history_queue.append((buys, sells))

        if len(self.all_history_queue) >= self.window:
            buys_, sells_ = self.all_history_queue.popleft()
            self.ups -= buys_
            self.downs -= sells_

        self._value = self.calculate()
        super(TnS, self).step(value=self._value)

    def calculate(self):
        nom = round(self.ups - self.downs, 6)
        denom = round(self.ups + self.downs, 6)
        if denom == 0.:
            return 0.
        elif nom == 0.:
            return 0.
        else:
            return nom / denom
