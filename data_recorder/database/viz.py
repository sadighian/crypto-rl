import matplotlib.cm as cm
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt


def plot_lob_overlay(data: pd.DataFrame, window=1, levels=range(15)) -> None:
    """
    Plot limit order book midpoint prices on the first axis, and the price levels
    within the LOB on a second axis, centered around the midpoint.

    :param data: LOB snapshot export in form of a DataFrame
    :param window: rolling look-back period to smooth LOB price levels
    :param levels: a list of levels to render in the plot
    :return: (void)
    """

    def ma(a, n=window):
        if isinstance(a, list) or isinstance(a, np.ndarray):
            a = pd.DataFrame(a)
        return a.rolling(n).mean().values

    midpoint_prices = data['midpoint'].values

    colors = cm.rainbow(np.linspace(0, 1, len(levels)))

    fig, ax1 = plt.subplots(figsize=(16, 6))
    ax1.set_title("Midpoint Prices vs. Stationary Limit Order Book Levels")

    ax1.plot(midpoint_prices, label='midpoint', color='b')

    ax2 = ax1.twinx()

    for c, level in zip(colors, levels):
        ax2.plot(ma(data['bid_distance_{}'.format(level)].values, n=window),
                 linestyle='--', label='Bid-{}'.format(level), color=c, alpha=0.5)
        ax2.plot(ma(data['ask_distance_{}'.format(level)].values, n=window),
                 linestyle='--', label='Ask-{}'.format(level), color=c, alpha=0.5)

    ax2.axhline(0, color='y', linestyle='--')
    ax2.set_ylabel('Level Differences', color='y')
    ax2.tick_params('y', colors='y')

    ax1.set_xlabel("Time step (each tick is 1-second)")
    ax1.set_ylabel("Price (USD)")
    fig.legend()
    plt.show()


def _get_transaction_plot_values(data: pd.DataFrame) -> (
        np.ndarray, np.ndarray, pd.DataFrame):
    """
    Helper function to prepare transaction data for plotting.

    :param data: LOB snapshot export in form of a DataFrame
    :return: bid, ask, and transaction data for plotting
    """
    nbbo = data[['midpoint', 'bid_distance_0', 'ask_distance_0']].values
    bids = nbbo[:, 0] * (nbbo[:, 1] + 1.)
    asks = nbbo[:, 0] * (nbbo[:, 2] + 1.)

    transactions = data[['buys', 'sells']].copy()
    transactions /= transactions
    transactions = transactions.fillna(0.)

    transactions['buys'] = asks * transactions['buys'].values
    transactions['sells'] = bids * transactions['sells'].values

    transactions.loc[
        (transactions['buys'] == 0.), ['buys']] = np.nan
    transactions.loc[
        (transactions['sells'] == 0.), ['sells']] = np.nan

    return bids, asks, transactions


def plot_transactions(data: pd.DataFrame) -> None:
    """
    Plot midpoint prices with buy and sell transactions dotted on the same plotting axis.

    :param data: LOB snapshot export in form of a DataFrame
    :return: (void)
    """
    bids, asks, transactions = _get_transaction_plot_values(data)

    fig, ax1 = plt.subplots(figsize=(16, 6))

    ax1.plot(bids, label='bids', color='g', alpha=0.5)
    ax1.plot(asks, label='asks', color='r', alpha=0.5)

    ax1.set_xlabel("Time step (each tick is 1-second)")
    ax1.set_ylabel("Price (USD)")
    ax1.set_title("Bid vs. Ask spread with Buy and Sell executions")

    x = list(range(transactions.shape[0]))
    ax1.scatter(x, transactions['buys'], label='buys', c='g')
    ax1.scatter(x, transactions['sells'], label='sells', c='r')

    fig.legend()
    plt.show()


def plot_lob_levels(data: pd.DataFrame, window: int = 1, levels: list = range(1, 15),
                    include_transactions: bool = True) -> None:
    """
    Plot limit order book midpoint prices on the first axis, and the percentage
    distances for each price level within the LOB on a second axis, centered around the
    midpoint.

    :param data: LOB snapshot export in form of a DataFrame
    :param window: rolling look-back period to smooth LOB price levels
    :param levels: a list of levels to render in the plot
    :param include_transactions: if TRUE, plot transactions
    :return: (void)
    """

    def ma(a, n=window):
        if isinstance(a, list) or isinstance(a, np.ndarray):
            a = pd.DataFrame(a)
        return a.rolling(n).mean().values

    midpoint_prices = data['midpoint'].values
    bids, asks, transactions = _get_transaction_plot_values(data)
    colors = cm.rainbow(np.linspace(0, 1, len(levels)))

    fig, ax1 = plt.subplots(figsize=(16, 6))
    ax1.set_title("Bid-Ask Prices vs. Fixed Limit Order Book Levels")

    ax1.plot(bids, label='bids', color='g', alpha=0.5)
    ax1.plot(asks, label='asks', color='r', alpha=0.5)

    for i, (c, level) in enumerate(zip(colors, levels)):
        bid_level_data = data['bid_distance_{}'.format(level)].values + 1
        ax1.plot(ma(bid_level_data * midpoint_prices, window), linestyle='--',
                 label='Bid-{}'.format(level), color=c, alpha=0.5)
        ask_level_data = data['ask_distance_{}'.format(level)].values + 1
        ax1.plot(ma(ask_level_data * midpoint_prices, window), linestyle='--',
                 label='Ask-{}'.format(level), color=c)

    if include_transactions:
        x = list(range(transactions.shape[0]))
        ax1.scatter(x, transactions['buys'], label='buys', c='g')
        ax1.scatter(x, transactions['sells'], label='sells', c='r')

    ax1.set_xlabel("Time step (each tick is 1-second)")
    ax1.set_ylabel("Price (USD)")
    fig.legend()
    plt.show()


def plot_order_arrivals(data: pd.DataFrame, level: int = 0) -> None:
    """
    Plot midpoint prices on the first axis, and OFI on the second access.

    :param data: LOB snapshot export in form of a DataFrame
    :param level: price level number to render in the plot
    :return: (void)
    """
    fig, ax1 = plt.subplots(figsize=(16, 6))

    bids, asks, transactions = _get_transaction_plot_values(data)
    ax1.plot(bids, label='bids', color='g', alpha=0.5)
    ax1.plot(asks, label='asks', color='r', alpha=0.5)
    x_axis = list(range(transactions.shape[0]))
    ax1.scatter(x_axis, transactions['buys'], label='buys', c='g')
    ax1.scatter(x_axis, transactions['sells'], label='sells', c='r')

    ax2 = ax1.twinx()
    cancel_arrivals = data['bid_cancel_notional_{}'.format(level)] - \
                      data['ask_cancel_notional_0']
    limit_arrivals = data['bid_limit_notional_{}'.format(level)] - \
                     data['ask_limit_notional_0']
    market_arrivals = data['bid_market_notional_{}'.format(level)] - \
                      data['ask_market_notional_0']

    ofi = limit_arrivals - cancel_arrivals - market_arrivals
    ax2.bar(x_axis, ofi, label='ofi #{}'.format(level), color='orange', alpha=0.6)
    ax2.axhline(0., color='black', alpha=0.2, linestyle='-.')

    ax1.set_xlabel("Time step (each tick is 1-second)")
    ax1.set_ylabel("Price (USD)")
    ax2.set_ylabel("Notional Value (USD)")
    ax1.set_title("Midpoint price vs. Order Arrival Notional Values")

    fig.legend()
    plt.show()
