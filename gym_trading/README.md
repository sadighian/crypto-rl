# Trading Gym
As of April 28th, 2019.

## 1. Overview
This package is my implementation of an HFT environment using a POMDP framework.

There are several utility classes within this module:
1. `broker.py` manages orders, executions, position inventories, and PnL 
calculations generate feature data, and export feature data to csv, and
2. `trading_gym.py` is my implementation of a HFT environment using 
GYM's POMDP framework.

## 2. Reward Structure
The environment is currently configured to return realized PnL as
the reward. Moreover, a reward is only returned if a position is
flattened.

Note: more reward structure options will be added soon.