

class TradeTracker(object):

    def __init__(self):
        self._notional = 0.
        self._count = 0

    def __str__(self):
        return 'TradeTracker: [notional={} | count={}]'.format(
            self._notional, self._count)

    @property
    def notional(self):
        return self._notional

    @property
    def count(self):
        return self._count

    def clear(self):
        self._notional = 0.
        self._count = 0

    def add(self, notional=100.):
        self._notional += notional
        self._count += 1

    def remove(self, notional=100.):
        self._notional -= notional
        self._count -= 1
