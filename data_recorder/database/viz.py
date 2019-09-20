import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import matplotlib.cm as cm


def plot_lob_overlay(data: pd.DataFrame, window=1, levels=range(15)):
    def ma(a, n=window):
        if isinstance(a, list) or isinstance(a, np.ndarray):
            a = pd.DataFrame(a)
        return a.rolling(n).mean().values

    midpoint_prices = data['coinbase_midpoint'].values

    colors = cm.rainbow(np.linspace(0, 1, len(levels)))

    fig, ax1 = plt.subplots(figsize=(16, 6))
    ax1.set_title("Midpoint Prices vs Stationary Limit Order Book Levels")

    ax1.plot(midpoint_prices, label='coinbase_midpoint', color='b')

    ax2 = ax1.twinx()

    for c, level in zip(colors, levels):
        ax2.plot(ma(data['coinbase_bid_distance_{}'.format(level)].values, n=window),
                 linestyle='--', label='Bid-{}'.format(level), color=c, alpha=0.5)
        ax2.plot(ma(data['coinbase_ask_distance_{}'.format(level)].values, n=window),
                 linestyle='--', label='Ask-{}'.format(level), color=c, alpha=0.5)

    ax2.axhline(0, color='y', linestyle='--')
    ax2.set_ylabel('Level Differences', color='y')
    ax2.tick_params('y', colors='y')

    ax1.set_xlabel("Timestep (each tick is half-second)")
    ax1.set_ylabel("Price (USD)")
    fig.legend()
    plt.show()


def _get_transaction_plot_values(data: pd.DataFrame):
    nbbo = data[['coinbase_midpoint', 'coinbase_bid_distance_0',
                 'coinbase_ask_distance_0']].values
    bids = nbbo[:, 0] * (nbbo[:, 1] + 1)
    asks = nbbo[:, 0] * (nbbo[:, 2] + 1)

    transactions = data[['coinbase_buys', 'coinbase_sells']].copy()
    transactions /= transactions
    transactions = transactions.fillna(0.)

    transactions['coinbase_buys'] = asks * transactions['coinbase_buys'].values
    transactions['coinbase_sells'] = bids * transactions['coinbase_sells'].values

    transactions.loc[
        (transactions['coinbase_buys'] == 0.), ['coinbase_buys']] = np.nan
    transactions.loc[
        (transactions['coinbase_sells'] == 0.), ['coinbase_sells']] = np.nan
    return bids, asks, transactions


def plot_transactions(data: pd.DataFrame):
    bids, asks, transactions = _get_transaction_plot_values(data)

    fig, ax1 = plt.subplots(figsize=(16, 6))

    ax1.plot(bids, label='bids', color='g', alpha=0.5)
    ax1.plot(asks, label='asks', color='r', alpha=0.5)

    ax1.set_xlabel("Timestep (each tick is half-second)")
    ax1.set_ylabel("Price (USD)")
    ax1.set_title("Bid vs Ask spread with Buy and Sell executions")

    x = list(range(transactions.shape[0]))
    ax1.scatter(x, transactions['coinbase_buys'], label='buys', c='g')
    ax1.scatter(x, transactions['coinbase_sells'], label='sells', c='r')

    fig.legend()
    plt.show()


def plot_lob_levels(data: pd.DataFrame, window=1, levels=range(1, 15),
                    plot_transactions=True):
    def ma(a, n=window):
        if isinstance(a, list) or isinstance(a, np.ndarray):
            a = pd.DataFrame(a)
        return a.rolling(n).mean().values

    midpoint_prices = data['coinbase_midpoint'].values
    bids, asks, transactions = _get_transaction_plot_values(data)
    colors = cm.rainbow(np.linspace(0, 1, len(levels)))

    fig, ax1 = plt.subplots(figsize=(16, 6))
    ax1.set_title("Bid-Ask Prices vs Fixed Limit Order Book Levels")

    ax1.plot(bids, label='bids', color='g', alpha=0.5)
    ax1.plot(asks, label='asks', color='r', alpha=0.5)

    for i, (c, level) in enumerate(zip(colors, levels)):
        bid_level_data = data['coinbase_bid_distance_{}'.format(level)].values + 1
        ax1.plot(ma(bid_level_data * midpoint_prices, window), linestyle='--',
                 label='Bid-{}'.format(level), color=c, alpha=0.5)
        ask_level_data = data['coinbase_ask_distance_{}'.format(level)].values + 1
        ax1.plot(ma(ask_level_data * midpoint_prices, window), linestyle='--',
                 label='Ask-{}'.format(level), color=c)

    if plot_transactions:
        x = list(range(transactions.shape[0]))
        ax1.scatter(x, transactions['coinbase_buys'], label='buys', c='g')
        ax1.scatter(x, transactions['coinbase_sells'], label='sells', c='r')

    ax1.set_xlabel("Timestep (each tick is half-second)")
    ax1.set_ylabel("Price (USD)")
    fig.legend()
    plt.show()


def plot_order_arrivals(data, level=0):
    fig, ax1 = plt.subplots(figsize=(16, 6))

    bids, asks, transactions = _get_transaction_plot_values(data)
    ax1.plot(bids, label='bids', color='g', alpha=0.5)
    ax1.plot(asks, label='asks', color='r', alpha=0.5)
    x_axis = list(range(transactions.shape[0]))
    ax1.scatter(x_axis, transactions['coinbase_buys'], label='buys', c='g')
    ax1.scatter(x_axis, transactions['coinbase_sells'], label='sells', c='r')

    ax2 = ax1.twinx()
    cancel_arrivals = data['coinbase_bid_cancel_notional_{}'.format(level)] - data[
        'coinbase_ask_cancel_notional_0']
    limit_arrivals = data['coinbase_bid_limit_notional_{}'.format(level)] - data[
        'coinbase_ask_limit_notional_0']
    market_arrivals = data['coinbase_bid_market_notional_{}'.format(level)] - data[
        'coinbase_ask_market_notional_0']

    ax2.bar(x_axis, cancel_arrivals, label='cancel arrivals {}'.format(level),
            color='red', alpha=0.6)
    ax2.bar(x_axis, limit_arrivals, label='limit_arrivals {}'.format(level),
            color='green', alpha=0.6)
    ax2.bar(x_axis, market_arrivals, label='market_arrivals {}'.format(level),
            color='purple', alpha=0.6)
    ax2.axhline(0., color='black', alpha=0.2, linestyle='-.')

    ax1.set_xlabel("Timestep (each tick is half-second)")
    ax1.set_ylabel("Price (USD)")
    ax2.set_ylabel("Notional Value (USD)")
    ax1.set_title("Midpoint price vs Order Arrival Notional Values")

    fig.legend()
    plt.show()
