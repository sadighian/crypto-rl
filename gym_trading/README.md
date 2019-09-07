# GYM_TRADING
As of September 07, 2019.

## Overview
This package is my implementation of an HFT environment extending a
POMDP framework from OpenAI.

## 1. Envs
Module containing various environment implementations.
- `price_jump.py` implementation where the agent uses **market orders**
  to trade crytocurrencies.
- `market_maker.py` implementation where the agent uses 
**limit orders** to trade cryptocurrencies

## 2. Indicators
Module containing indicators that are implemented within environments. 

## 3. Utils
Module containing utility classes for the `gym_trading` module.
- `broker.py` manages orders, executions, position inventories, and PnL
  calculations for `price_jump.py`
- `broker_mm.py` manages orders, executions, position inventories, and
  PnL calculations for `market_maker.py`
- `render_env.py` render's midpoint price data as the agent steps
  through the environment

## 4. Tests
Module containing test cases for `gym_trading`'s modules.
