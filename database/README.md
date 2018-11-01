# Database
As of November 1st, 2018.

## 1. Overview
The `database` module contains the wrapper classes for storing and accessing
tick data from the `Arctic Tick Store`.

## 2. Classes

### 2.1 Database
This is a wrapper class used for storing streaming tick data into the `Arctic Tick Store`. 
The `new_tick()` method is implemented in both `bifinex_connector` and 
`coinbase_connector` projects.