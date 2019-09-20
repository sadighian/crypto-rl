import numpy as np


def load_ema(alpha=None):
    """
    Set exponential moving average smoother.
    :param alpha: decay rate for EMA
    :return: (var) EMA
    """
    if alpha is None:
        # print("EMA smoothing DISABLED")
        return None
    elif isinstance(alpha, float):
        # print("EMA smoothing ENABLED: {}".format(alpha))
        return ExponentialMovingAverage(alpha=alpha)
    elif isinstance(alpha, list):
        # print("EMA smoothing ENABLED: {}".format(alpha))
        return [ExponentialMovingAverage(alpha=a) for a in alpha]
    else:
        print("_load_ema() --> unknown alpha type: {}".format(type(alpha)))
        return None


def apply_ema_all_data(ema, data: np.array):
    """
    Apply exponential moving average to entire data set in a single batch
    :param ema: EMA handler; if None, no EMA is applied
    :param data: data set to smooth
    :return: smoothed data set, if ema is provided
    """
    smoothed_data = []
    if ema is None:
        return data
    elif isinstance(ema, ExponentialMovingAverage):
        for row in data:
            ema.step(value=row)
            smoothed_data.append(ema.value)
        return np.asarray(smoothed_data, dtype=np.float32)
    elif isinstance(ema, list):
        for row in data:
            tmp_row = []
            for e in ema:
                e.step(value=row)
                tmp_row.append(e.value)
            smoothed_data.append(tmp_row)
        return np.asarray(smoothed_data, dtype=np.float32).reshape(
            data.shape[0], -1)
    else:
        print("_apply_ema() --> unknown ema type: {}".format(type(ema)))
        return None


class ExponentialMovingAverage(object):

    def __init__(self, alpha: float):
        """
        Calculate Exponential moving average in O(1) time
        :param alpha: decay factor, usually between 0.9 and 0.9999
        """
        self.alpha = alpha
        self._value = None

    def __str__(self):
        return 'ExponentialMovingAverage: [ alpha={} | value={} ]'.format(
            self.alpha, self._value)

    def step(self, value: float):
        """
        Update EMA at every time step
        :param value: price at current time step
        :return: (void)
        """
        if self._value is None:
            self._value = value
            return

        self._value = (1. - self.alpha) * value + self.alpha * self._value

    @property
    def value(self):
        """
        EMA value of data
        :return: (float) EMA smoothed value
        """
        return self._value

    def reset(self):
        """
        Reset EMA
        :return: (void)
        """
        self._value = None
