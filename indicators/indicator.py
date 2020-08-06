from abc import ABC, abstractmethod
from collections import deque
from typing import List, Tuple, Union

from configurations import INDICATOR_WINDOW
from indicators.ema import ExponentialMovingAverage, load_ema


class Indicator(ABC):

    def __init__(self, label: str,
                 window: Union[int, None] = INDICATOR_WINDOW[0],
                 alpha: Union[List[float], float, None] = None):
        """
        Indicator constructor.

        :param window: (int) rolling window used for indicators
        :param alpha: (float) decay rate for EMA; if NONE, raw values returned
        """
        self._label = f"{label}_{window}"
        self.window = window
        if self.window is not None:
            self.all_history_queue = deque(maxlen=self.window + 1)
        else:
            self.all_history_queue = deque(maxlen=2)
        self.ema = load_ema(alpha=alpha)
        self._value = 0.

    def __str__(self):
        return f'Indicator.base() [ window={self.window}, ' \
               f'all_history_queue={self.all_history_queue}, ema={self.ema} ]'

    @abstractmethod
    def reset(self) -> None:
        """
        Clear values in indicator cache.

        :return: (void)
        """
        self._value = 0.
        self.all_history_queue.clear()

    @abstractmethod
    def step(self, **kwargs) -> None:
        """
        Update indicator with steps from the environment.

        :param kwargs: data values passed to indicators
        :return: (void)
        """
        if self.ema is None:
            pass
        elif isinstance(self.ema, ExponentialMovingAverage):
            self.ema.step(**kwargs)
        elif isinstance(self.ema, list):
            for ema in self.ema:
                ema.step(**kwargs)
        else:
            pass

    @abstractmethod
    def calculate(self, *args, **kwargs) -> float:
        """
        Calculate indicator value.

        :return: (float) value of indicator
        """
        pass

    @property
    def value(self) -> Union[List[float], float]:
        """
        Get indicator value for the current time step.

        :return: (scalar float)
        """
        if self.ema is None:
            return self._value
        elif isinstance(self.ema, ExponentialMovingAverage):
            return self.ema.value
        elif isinstance(self.ema, list):
            return [ema.value for ema in self.ema]
        else:
            return 0.

    @property
    def label(self) -> Union[List[str], str]:
        """
        Get indicator value for the current time step.

        :return: (scalar float)
        """
        if self.ema is None:
            return self._label
        elif isinstance(self.ema, ExponentialMovingAverage):
            return f"{self._label}_{self.ema.alpha}"
        elif isinstance(self.ema, list):
            return [f"{self._label}_{ema.alpha}" for ema in self.ema]
        else:
            raise ValueError(f"Error: EMA provided not valid --> {self.ema}")

    @property
    def raw_value(self) -> float:
        """
        Guaranteed raw value, if EMA is enabled.

        :return: (float) raw indicator value
        """
        return self._value

    @staticmethod
    def safe_divide(nom: float, denom: float) -> float:
        """
        Safely perform divisions without throwing an 'divide by zero' exception.

        :param nom: nominator
        :param denom: denominator
        :return: value
        """
        if denom == 0.:
            return 0.
        elif nom == 0.:
            return 0.
        else:
            return nom / denom


class IndicatorManager(object):
    __slots__ = ['indicators']

    def __init__(self):
        """
        Wrapper class to manage multiple indicators at the same time
        (e.g., window size stacking)

        # :param smooth_values: if TRUE, values returned are EMA smoothed, otherwise raw
        #     values indicator values
        """
        self.indicators = list()

    def get_labels(self) -> list:
        """
        Get labels for each indicator being managed.

        :return: List of label names
        """
        # return [label[0] for label in self.indicators]
        labels = []
        for label, indicator in self.indicators:
            indicator_label = indicator.label
            if isinstance(indicator_label, list):
                labels.extend(indicator_label)
            else:
                labels.append(indicator_label)
        return labels

    def add(self, name_and_indicator: Tuple[str, Union[Indicator, ExponentialMovingAverage]]) \
            -> None:
        """
        Add indicator to the list to be managed.

        :param name_and_indicator: tuple(name, indicator)
        :return: (void)
        """
        self.indicators.append(name_and_indicator)

    def delete(self, index: Union[int, None]) -> None:
        """
        Delete an indicator from the manager.

        :param index: index to delete (int or str)
        :return: (void)
        """
        if isinstance(index, int):
            del self.indicators[index]
        else:
            self.indicators.remove(index)

    def pop(self, index: Union[int, None]) -> Union[float, None]:
        """
        Pop indicator from manager.

        :param index: (int) index of indicator to pop
        :return: (name, indicator)
        """
        if index is not None:
            return self.indicators.pop(index)
        else:
            return self.indicators.pop()

    def step(self, **kwargs) -> None:
        """
        Update indicator with new step through environment.

        :param kwargs: Data passed to indicator for the update
        :return:
        """
        for (name, indicator) in self.indicators:
            indicator.step(**kwargs)

    def reset(self) -> None:
        """
        Reset all indicators being managed.

        :return: (void)
        """
        for (name, indicator) in self.indicators:
            indicator.reset()

    def get_value(self) -> List[float]:
        """
        Get all indicator values in the manager's inventory.

        :return: (list of floats) Indicator values for current time step
        """
        values = []
        for name, indicator in self.indicators:
            indicator_value = indicator.value
            if isinstance(indicator_value, list):
                values.extend(indicator_value)
            else:
                values.append(indicator_value)
        return values
