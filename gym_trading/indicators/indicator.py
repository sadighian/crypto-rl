from abc import ABC, abstractmethod
from configurations.configs import INDICATOR_WINDOW
from collections import deque


class Indicator(ABC):

    def __init__(self, window=INDICATOR_WINDOW):
        """
        Indicator constructor
        :param window: (int) rolling window used for indicators
        """
        self.window = window
        self.all_history_queue = deque(maxlen=self.window)

    def __str__(self):
        return 'Indicator.base() [window={}, all_history_queue={}]'.format(
            self.window, self.all_history_queue)

    @abstractmethod
    def reset(self):
        """
        Clear values in indicator cache
        :return: (void)
        """
        pass

    @abstractmethod
    def step(self, **kwargs):
        """
        Update indicator with steps from the environment
        :param kwargs: data values passed to indicators
        :return: (void)
        """
        pass

    @abstractmethod
    def get_value(self):
        """
        Get indicator value for the current time step
        :return: (scalar float)
        """
        pass


class IndicatorManager(object):

    def __init__(self):
        """
        IndicatorManager constructor
        """
        self.indicators = []

    def add(self, name_and_indicator):
        """
        Add indicator to the list to be managed
        :param name_and_indicator: tuple(name, indicator)
        :return: (void)
        """
        self.indicators.append(name_and_indicator)

    def delete(self, index):
        """
        Delete an indicator from the manager
        :param index: index to delete (int or str)
        :return: (void)
        """
        if isinstance(index, int):
            del self.indicators[index]
        else:
            self.indicators.remove(index)

    def pop(self, index=None):
        """
        Pop indicator from manager
        :param index: (int) index of indicator to pop
        :return: (name, indicator)
        """
        if index:
            return self.indicators.pop(index=index)
        else:
            return self.indicators.pop()

    def step(self, **kwargs):
        """
        Update indicator with new step through environment
        :param kwargs: Data passed to indicator for the update
        :return:
        """
        for (name, indicator) in self.indicators:
            getattr(indicator, 'step')(**kwargs)

    def reset(self):
        """
        Reset all indicators being managed
        :return: (void)
        """
        for (name, indicator) in self.indicators:
            getattr(indicator, 'reset')()

    def get_value(self):
        """
        Get all indicator values in the manager's inventory
        :return: (list of floats) Indicator values for current time step
        """
        return [getattr(indicator, 'get_value')()
                for (name, indicator) in self.indicators]
