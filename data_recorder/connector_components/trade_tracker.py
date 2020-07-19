class TradeTracker(object):

    def __init__(self):
        """
        Constructor.
        """
        self._notional = 0.
        self._count = 0

    def __str__(self):
        return 'TradeTracker: [notional={} | count={}]'.format(
            self._notional, self._count)

    @property
    def notional(self) -> float:
        """
        Total notional value of transactions since last TradeTracker.clear().

        Example:
            notional = price * quantity

        :return: notional value
        """
        return self._notional

    @property
    def count(self) -> int:
        """
        Total number of transactions since last TradeTracker.clear().

        :return: count of transactions
        """
        return self._count

    def clear(self) -> None:
        """
        Reset the trade values for notional and count to zero (intended to be called
        every time step).

        :return: (void)
        """
        self._notional = 0.
        self._count = 0

    def add(self, notional: float) -> None:
        """
        Add a trade's notional value to the cumulative sum and counts of transactions
        since last TradeTracker.clear().

        :param notional: notional value of transaction
        :return: (void)
        """
        self._notional += notional
        self._count += 1

    def remove(self, notional: float) -> None:
        """
        Remove a trade's notional value from the cumulative sum and counts of
        transactions since last
            TradeTracker.clear().

        :param notional: notional value of transaction
        :return: (void)
        """
        self._notional -= notional
        self._count -= 1
