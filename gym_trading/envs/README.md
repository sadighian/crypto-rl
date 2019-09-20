# Environments
As of September 20, 2019.

## 1. Overview
Each python file is an extension to OpenAI's GYM module 
with all mandatory abstract methods are implemented.

## 2. Environments

### 2.1 base_env.py
- Base class environment to serve as a platform for extending
- Includes all repetitive functions required by environments: (1) data
  loading and preprocessing, (2) broker to act as counterparty, and (3)
  other attributes
- `rewards` can be derived via 4 different approaches:
  1.   'trade_completion' : reward is generated per trade's round trip
  2.   'continuous_total_pnl' : change in realized & unrealized pnl
       between time steps
  3.   'continuous_realized_pnl' : change in realized pnl between time
       steps
  4.   'continuous_unrealized_pnl' : change in unrealized pnl between
       time steps
  5.   'normed' : refer to https://arxiv.org/abs/1804.04216v1
  6.   'div' : reward is generated per trade's round trip divided by
- `observation space` can be normalized via: 
    1.  z-score
    2. min-max (e.g., range [0,1])
- The position management and PnL calculator are handled by the
  `../broker.py` class in FIFO order
- The historical data is loaded using `../data_recorder/database/simulator.py`
 class

### 2.2 price_jump.py
- This environment is designed for MARKET orders only with the objective
  being able to identify a "price jump"
- Rewards in this environment are realized PnL in FIFO order 
(i.e., current midpoint)
- The `../agent/dqn.py` Agent implements this class
 
### 2.3 market_maker.py
- This environment is designed for LIMIT orders only with the objective
  being able profit from market making
- Rewards in this environment are realized PnL in FIFO order
- If there are partial executions, the average execution price is used
  to determine PnL
 - The `../agent/dqn.py` Agent implements this class