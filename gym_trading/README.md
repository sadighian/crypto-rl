# GYM_TRADING
As of May 17th, 2019.

## 1. Overview
This package is my implementation of an HFT environment using a POMDP framework.

The classes consist of environments and their respective utility classes:
1. `broker.py` and `broker2.py` manages orders, executions, 
position inventories, and PnL calculations generate feature 
data, and export feature data to a csv,
2. `price_jump.py` and `market_maker.py` are environment implementations
using the OpenAI Gym framework. These environments use the recorded limit
order book data as state space representation.