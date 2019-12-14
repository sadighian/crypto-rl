# Connector Components
As of September 09, 2019.

## 1. Overview
The `connector_components` module contains the base classes for 
connecting to crypto exchanges. Each base class is overriden in the 
following modules:
- `bitfinex_connector/`
- `coinbase_connector/`
- `bitmex_connector/`


## 2. Classes

### 2.1 Book
This class is responsible for maintaining the inventory all of the buy
**or** sell orders through implementing the `./price_level.py` class.

### 2.2 Client
This class is responsible for creating WebSocket connections to an 
exchange endpoint.

### 2.3 Order Book
This class is responsible for implementing the `./book.py` class for 
both buy and sell orders, thus making an actual order book.

### 2.4 Price Level
This class is responsible for keeping track order inventories at a given
price. This class is instantiated for every price level in the limit
order book. Order flow arrival attributes are reset each time a LOB
snapshot is taken.

### 2.5 Trade Tracker
This class is responsible for keeping track of the time and sales
transactional data. Order flow arrival attributes are reset each time a
LOB snapshot is taken.