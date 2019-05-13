# GYM_TRADING
As of May 13th, 2019.

## 1. Overview
This package is my implementation of an HFT environment using a POMDP framework.

There are several utility classes within this module:
1. `broker.py` and `broker2.py` manages orders, executions, 
position inventories, and PnL calculations generate feature 
data, and export feature data to csv, and
2. `price_jump.py` and `market_maker.py` are environment implementations
using the OpenAI Gym framework. These environments use the recorded limit
order book data for the observation state space.

Note: more reward structure options will be added soon.