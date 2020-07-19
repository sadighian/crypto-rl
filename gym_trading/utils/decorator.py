import time
from functools import wraps


def debugging(func):
    """
    Decorator for debugging inputs into a function.
    :param func: function
    :return: function
    """

    @wraps(func)
    def f(*args, **kwargs):
        if kwargs != {}:
            _kwargs = '\n'.join(["--{}\t=\t{}".format(k, v) for k, v in kwargs.items()])
        else:
            _kwargs = 'None'
        print('-' * 50)
        print("{}(\nkwargs::\n   {}\n)".format(func.__name__, _kwargs))
        print('-' * 50)
        return func(*args, **kwargs)

    return f


def print_time(func):
    """
    Decorator for timing function execution duration.

    :param func: function
    :return: wrapped function
    """

    @wraps(func)
    def f(*args, **kwargs):
        start_time = time.time()
        output = func(*args, **kwargs)
        elapsed = time.time() - start_time
        print("{} completed in {:.4f} seconds.".format(func.__name__, elapsed))
        return output

    return f
