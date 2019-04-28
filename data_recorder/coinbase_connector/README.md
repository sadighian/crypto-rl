# Coinbase Pro Connector
As of March 04, 2019.

## 1. Overview
The Coinbase connector consists of three classes:
1. `coinbase_book.py` which is the gdax implementation of `./connector_components/book.py`
2. `coinbase_orderbook.py` which is the gdax implementation of `./connector_components/orderbook.py`
3. `coinbase_client.py` which is the gdax implementation of `./connector_components/client.py`

## 2. Subscriptions
- WebSocket connections are made asynchronously using the `websockets` module
- Full order book data subscriptions are made by using the `subscribe()` method 
from `coinbase_client.py`
- Orderbook snapshots are made using a GET call using the `requests` module
- All websocket subscriptions pass incoming messages into a `multiprocessing.Queue()` and 
to be processed by a separate thread

## 3. Data Consumption Rules
1. Filter out messages with `type` = `received` to save time
2. Normalize incoming data messages from strings to numbers, such as `floats()`
3. Pass normalized messages to the `orderbook.new_tick()` method to update the limit order book
4. If the websocket feed looses connection, try to re-subscribe again

## 4. Appendix 
Link to official Coinbase documentation: https://docs.pro.coinbase.com/
