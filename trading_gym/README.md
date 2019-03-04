# Trading Gym
As of March 04, 2019.

## 1. Overview
This package is my implementation of an HFT environment using a POMDP framework.


There are several utility classes within this module:
1. `broker.py` manages orders, executions, position inventories, and PnL calculations;
2. `agent.py` is a wrapper class implementing `trading_gym.py` and `keras-rl.dqn_agent`;
3. `simulator.py` is a utility class to query Arctic (database), 
generate feature data, and export feature data to csv; and,
4. `trading_gym.py` is my implementation of a HFT environment using GYM's POMDP framework.
