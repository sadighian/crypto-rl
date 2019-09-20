from abc import ABC, abstractmethod
from configurations.configs import INDICATOR_WINDOW
from collections import deque
from indicators import load_ema, ExponentialMovingAverage


class Indicator(ABC):

    def __init__(self, window=INDICATOR_WINDOW, alpha=None):
        """
        Indicator constructor
        :param window: (int) rolling window used for indicators
        :param alpha: (float) decay rate for EMA; if NONE, raw values returned
        """
        self.window = window
        self.all_history_queue = deque(maxlen=self.window)
        self.ema = load_ema(alpha=alpha)
        self._value = None

    def __str__(self):
        return 'Indicator.base() [window={}, all_history_queue={}]'.format(
            self.window, self.all_history_queue)

    @abstractmethod
    def reset(self):
        """
        Clear values in indicator cache
        :return: (void)
        """
        self._value = None

    @abstractmethod
    def step(self, **kwargs):
        """
        Update indicator with steps from the environment
        :param kwargs: data values passed to indicators
        :return: (void)
        """
        if isinstance(self.ema, ExponentialMovingAverage):
            self.ema.step(**kwargs)
        elif isinstance(self.ema, list):
            for ema in self.ema:
                ema.step(**kwargs)
        elif self.ema is None:
            pass
        else:
            pass

    @abstractmethod
    def calculate(self):
        """
        Calculate indicator value
        :return: (float) value of indicator
        """
        pass

    @property
    def value(self):
        """
        Get indicator value for the current time step
        :return: (scalar float)
        """
        if isinstance(self.ema, ExponentialMovingAverage):
            return self.ema.value
        elif isinstance(self.ema, list):
            return [ema.value for ema in self.ema]
        elif self.ema is None:
            return self._value
        else:
            return 0.


class IndicatorManager(object):

    def __init__(self):
        """
        Wrapper class to manage multiple indicators at the same time
        (e.g., window size stacking)
        # :param smooth_values: if TRUE, values returned are EMA smoothed, otherwise raw
        #     values indicator values
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
        return [getattr(indicator, 'value')
                for (name, indicator) in self.indicators]
