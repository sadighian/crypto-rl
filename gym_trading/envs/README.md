# Environments
As of December 12, 2019.

## 1. Overview
Each python file is an extension to OpenAI's GYM module 
with all mandatory abstract methods are implemented.

## 2. Environments

### 2.1 base_env.py
- Base class environment to serve as a platform for extending
- Includes all repetitive functions required by environments: (1) data
  loading and pre-processing, (2) broker to act as counter-party, and (3)
  other attributes
- `rewards` can be derived via several different approaches:
    1) 'default' --> inventory count * change in midpoint price returns
    2) 'default_with_fills' --> inventory count * change in midpoint price returns + closed trade
     PnL
    3) 'realized_pnl' --> change in realized pnl between time steps
    4) 'differential_sharpe_ratio' -->
        http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.1.7210&rep=rep1&type=pdf
    5) 'asymmetrical' --> extended version of *default* and enhanced with a
        reward for being filled above or below midpoint, and returns only
        negative rewards for Unrealized PnL to discourage long-term
        speculation.
    6) 'trade_completion' --> reward is generated per trade's round trip
       
- `observation space` is normalized via z-score; outliers above +/-10 are clipped.
- The position management and PnL calculator are handled by the
  `../broker.py` class in FIFO order
- The historical data is loaded using `../gym_trading/utils/data_pipeline.py`
 class

### 2.2 trend_following.py
- This environment is designed for MARKET orders only with the objective
  being able to identify a "price jump" or directional movement
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