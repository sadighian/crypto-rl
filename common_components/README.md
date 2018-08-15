# Common Components
As of August 14th, 2018.

## 1. Overview
The `common_components` module contains the base classes for the overall project. 
Each base class is overriden in the `bitfinex_connector` and `gdax_connector` projects. 

## 2. Base Classes

### 2.1 Book
This class is responsible for maintaining the inventory all of the buy **or** sell orders.

### 2.2 Order Book
This class is responsible for implementing the `./book.py` class for both buy and sell orders, 
thus making an actual order book.

### 2.3 Client
This class is responsible for creating websocket connections to an exchange endpoint.

## 3. Other Utility Classes

### 3.1 Configs
This class is the single source of all configurations within the project.

### 3.2 Database
This class is the interface with the `Arctic Tick Store` database and is 
implemented both `bitfinex_orderbook.py` and `gdax_orderbook.py` classes