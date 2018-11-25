# Database
As of November 25th, 2018.

## 1. Overview
The `database` module contains the wrapper class for storing tick data from the `Arctic Tick Store`.

**Note** the `../simulator.py` class is used to query and reconstruct the limit order book.

## 2. Classes

### 2.1 Database
This is a wrapper class used for storing streaming tick data into the `Arctic Tick Store`. 
The `new_tick()` method is implemented in both `bifinex_connector` and 
`coinbase_connector` projects.
