# GYM_TRADING
As of December 12, 2019.

## Overview
This package is my implementation of an HFT environment extending a
POMDP framework from OpenAI.

## 1. Envs
Module containing various environment implementations.
- `trend_following.py` implementation where the agent uses **market orders**
  to trade crytocurrencies.
- `market_maker.py` implementation where the agent uses **limit orders**
  to trade cryptocurrencies

## 2. Utils
Module containing utility classes for the `gym_trading` module.
- `broker.py`, `position.py`, and `order.py` manages orders &
  executions, position inventories, and PnL calculations for `envs`
- `render_env.py` renders midpoint price data as the agent steps
  through the environment
- `plot_history.py` renders environment observations and PnL
- `reward.py` contains the reward functions
- `statistics` contains trackers for risk, rewards, etc.

## 3. Tests
Module containing test cases for `gym_trading`'s modules.
