# Environments
As of September 07, 2019.

## 1. Overview
Each python file is an extension to OpenAI's GYM module 
with all mandatory abstract methods are implemented.

## 2. Environments

### 2.1 price_jump.py
- This environment is designed for market orders only with the
objective being able to identify a "price jump"
- Rewards in this environment are realized PnL
- Each time the Agent creates a new market order, 
the "stop loss" and "take profit" levels are a predefined 
distance from the position entry price 
(i.e., current midpoint)
- The position management and PnL calculator are handled by 
the `../broker.py` class
 - The historical data is loaded using `../data_recorder/database/simulator.py`
 class
 - The `../agent/dqn.py` Agent implements this class
 
### 2.2 market_maker.py
- This environment is designed for limit orders only with the
objective being able profit from market making
- Rewards in this environment are realized PnL in FIFO order
- Currently, partial executions are not taken into consideration
- The position management and PnL calculator are handled by the
  `../broker_mm.py` class
 - The historical data is loaded using `../data_recorder/database/simulator.py`
 class
 - The `../agent/dqn.py` Agent implements this class