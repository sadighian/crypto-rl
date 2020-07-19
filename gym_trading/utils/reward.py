import numpy as np


def default(inventory_count: int, midpoint_change: float) -> float:
    """
    Default reward type for environments, which is derived from PnL and order quantity.

    The inputs are as follows:
        (1) Change in exposure value between time steps, in dollar terms; and,
        (2) Realized PnL from a open order being filled between time steps,
            in dollar terms.

    :param inventory_count: TRUE if long order is filled within same time step
    :param midpoint_change: percentage change in midpoint price
    :return: reward
    """
    reward = inventory_count * midpoint_change
    return reward


def default_with_fills(inventory_count: int, midpoint_change: float, step_pnl: float) -> float:
    """
    Same as Default reward type for environments, but includes PnL from closing positions.

    The inputs are as follows:
        (1) Change in exposure value between time steps, in dollar terms; and,
        (2) Realized PnL from a open order being filled between time steps,
            in dollar terms.

    :param inventory_count: TRUE if long order is filled within same time step
    :param midpoint_change: percentage change in midpoint price
    :param step_pnl: limit order pnl
    :return: reward
    """
    reward = (inventory_count * midpoint_change) + step_pnl
    return reward


def realized_pnl(current_pnl: float, last_pnl: float) -> float:
    """
    Only provide reward signal when a trade is closed (round-trip).

    :param current_pnl: Realized PnL at current time step
    :param last_pnl: Realized PnL at former time step
    :return: reward
    """
    reward = current_pnl - last_pnl
    return reward


def differential_sharpe_ratio(R_t: float, A_tm1: float, B_tm1: float,
                              eta: float = 0.01) -> (float, float, float):
    """
    Method to calculate Differential Sharpe Ratio online.

    Source 1: http://www.cs.cmu.edu/afs/cs/project/link-3/lafferty/www/ml-stat-www/moody.pdf
    Source 2: http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.87.8437&rep=rep1&type
    =pdf

    :param R_t: reward from current time step (midpoint price change a.k.a. 'price returns')
    :param A_tm1: A from previous time step
    :param B_tm1: B form previous time step
    :param eta: discount rate (same as EMA's alpha)
    :return: (tuple) reward, A_t, and B_t
    """
    if R_t == 0.:
        return 0., A_tm1, B_tm1

    reward = 0.

    A_delta = R_t - A_tm1
    B_delta = R_t ** 2 - B_tm1

    A_t = A_tm1 + eta * A_delta
    B_t = B_tm1 + eta * B_delta

    nominator = B_tm1 * A_delta - (0.5 * A_tm1 * B_delta)
    denominator = (B_tm1 - A_tm1 ** 2) ** 1.5

    if np.isnan(nominator):
        return reward, A_t, B_t
    elif nominator == 0.:
        return reward, A_t, B_t
    elif denominator == 0.:
        return reward, A_t, B_t

    # scale down the feedback signal by 1/100th to avoid large spikes
    reward = (nominator / denominator) * 0.01

    return reward, A_t, B_t


def asymmetrical(inventory_count: int, midpoint_change: float, half_spread_pct: float,
                 long_filled: bool, short_filled: bool, step_pnl: float,
                 dampening: float = 0.6) -> float:
    """
    Asymmetrical reward type for environments, which is derived from percentage
    changes and notional values.

    The inputs are as follows:
        (1) Change in exposure value between time steps, in percentage terms; and,
        (2) Realized PnL from a open order being filled between time steps,
            in percentage.

    :param inventory_count: Number of open positions
    :param midpoint_change: Percentage change of midpoint between steps
    :param half_spread_pct: Percentage distance from bid/ask to midpoint
    :param long_filled: TRUE if long order is filled within same time step
    :param short_filled: TRUE if short order is filled within same time step
    :param step_pnl: limit order pnl and any penalties for bad actions
    :param dampening: discount factor towards pnl change between time steps
    :return: (float) reward
    """
    exposure_change = inventory_count * midpoint_change
    fill_reward = 0.

    if long_filled:
        fill_reward += half_spread_pct
    if short_filled:
        fill_reward += half_spread_pct

    reward = fill_reward + min(0., exposure_change * dampening)

    if long_filled or short_filled:
        reward += step_pnl

    return reward


def trade_completion(step_pnl: float, market_order_fee: float,
                     profit_ratio: float = 2.) -> float:
    """
    Alternate approach for reward calculation which places greater importance on
    trades that have returned at least a 1:1 profit-to-loss ratio after
    transaction fees.

    :param step_pnl: limit order pnl and any penalties for bad actions
    :param market_order_fee: transaction fee for market orders
    :param profit_ratio: minimum profit-to-risk ratio to earn '1' point (e,g., 2x)
    :return: reward
    """
    reward = 0.0

    if step_pnl > market_order_fee * profit_ratio:  # e.g.,  2:1 profit to loss ratio
        reward += 1.0
    elif step_pnl > 0.0:  # Not a 2:1 PL ratio, but still a positive return
        reward += step_pnl
    elif step_pnl < -market_order_fee:  # Loss is more than the transaction fee
        reward -= 1.0
    else:  # Loss is less than the transaction fee and negative
        reward += step_pnl

    return reward
