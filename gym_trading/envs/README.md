# Environments
As of April 28th, 2019.

## Overview
The `trading_gym.py` is an extension to OpenAI's GYM module. All
abstract methods are implemented. 
- The position management and PnL calculator are handled by 
the `../broker.py` class.
 - The historical data is loaded using `../data_recorder/database/simulator.py`
 class
 - The `../agent/dqn.py` Agent extends this class.