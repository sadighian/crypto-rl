# GDAX Connector
As of August 14th, 2018.

## 1. Overview
The GDAX connector consists of three classes:
1. `gdax_book.py` which is the gdax implementation of `./common_components/book.py`
2. `gdax_orderbook.py` which is the gdax implementation of `./common_components/orderbook.py`
3. `gdax_client.py` which is the gdax implementation of `./common_components/client.py`

## 2. Subscriptions
- WebSocket connections are made asynchronously using the `websockets` module
- Full order book data subscriptions are made by using the `subscribe()` method 
from `gdax_client.py`
- Orderbook snapshots are made using a GET call using the `requests` module
- All websocket subscriptions pass incoming messages into a `multiprocessing.Queue()` and 
to be processed by a separate thread

## 3. Data Consumption Rules
1. Filter out messages with `type` = `received` to save time
2. Normalize incoming data messages from strings to numbers, such as `floats()`
3. Pass normalized messages to the `orderbook.new_tick()` method to update the limit order book
4. If the websocket feed looses connection, try to re-subscribe again

## 4. Appendix 
Link to official GDAX documentation: https://docs.gdax.com/