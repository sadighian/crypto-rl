class TradeStatistics(object):
    """
    Class for storing order metrics performed by the agent.
    """

    def __init__(self):
        """
        Instantiate the class.
        """
        self.orders_executed = 0
        self.orders_placed = 0
        self.orders_updated = 0
        self.market_orders = 0

    def __str__(self):
        return ('TradeStatistics:\n'
                'orders_executed = \t{}\n'
                'orders_placed = \t{}\n'
                'orders_updated = \t{}\n'
                'market_orders = \t{}').format(
            self.orders_executed,
            self.orders_placed,
            self.orders_updated,
            self.market_orders
        )

    def reset(self) -> None:
        """
        Reset all trackers.

        :return: (void)
        """
        self.orders_executed = 0
        self.orders_placed = 0
        self.orders_updated = 0
        self.market_orders = 0


class ExperimentStatistics(object):

    def __init__(self):
        """
        Instantiate the class.
        """
        self.reward = 0.
        self.number_of_episodes = 0

    def __str__(self):
        return 'ExperimentStatistics:\nreward\t=\t{:.4f}'.format(self.reward) + \
               '\nNumber of Episodes\t=\t{:.4f}'.format(self.number_of_episodes)

    def reset(self) -> None:
        """
        Reset all trackers.

        :return: (void)
        """
        self.reward = 0.
