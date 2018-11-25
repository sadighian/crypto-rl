# Connector Components
As of November 25th, 2018.

## 1. Overview
The `connector_components` module contains the base classes for connecting to crypto exchanges. 
Each base class is overriden in the `bitfinex_connector` and `coinbase_connector` projects. 

## 2. Base Classes

### 2.1 Book
This class is responsible for maintaining the inventory all of the buy **or** sell orders.

### 2.2 Order Book
This class is responsible for implementing the `./book.py` class for both buy and sell orders, 
thus making an actual order book.

### 2.3 Client
This class is responsible for creating websocket connections to an exchange endpoint.