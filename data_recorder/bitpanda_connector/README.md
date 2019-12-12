# Bitpanda Connector
As of December 12, 2019.

## 1. Overview
The Bitpanda connector consists of three classes:
1. `bitpanda_book.py` which is the Bitpanda implementation of `./connector_components/book.py`
2. `bitpanda_orderbook.py` which is the Bitpanda implementation of `./connector_components/orderbook.py`
3. `bitpanda_client.py` which is the Bitpanda implementation of `./connector_components/client.py`

## 2. Subscriptions
- WebSocket connections are made asynchronously with the `websockets` module
- Raw order book & trades data subscriptions are made by using the `subscribe()` method 
from `bitpanda_client.py`
- All websocket subscriptions pass incoming messages into a `multiprocessing.Queue()` and 
to be processed by a separate thread

## 3. Data Consumption Rules
1. Normalize incoming data messages from strings to numbers, such as `floats()`
2. Pass normalized messages to the `orderbook.new_tick()` method to update the limit order book
3. If the websocket feed looses connection, try to re-subscribe again

## 4. Appendix 
Link to official Bitpanda documentation: https://developers.bitpanda.com/exchange/
