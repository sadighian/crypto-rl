# Multiprocessing Crypto Recorder
As of June 23th, 2018.
## 1. Purpose
Application designed to subscribe and insert_record the
full limit order book data from **GDAX** and **Bitfinex** into a MongoDB 
for reinforcement learning simulations.

## 2. Scope
Application is intended to be used to insert_record limit order book data for 
reinforcement learning modeling. Currently, there is no functionality 
developed to place order and actually trade.

## 3. Dependencies
- abc
- datetime
- pymongo
- time
- os
- ujson or json
- threading
- SortedDict
- websockets
- requests
- multiprocessing
- asyncio

## 4. Design Pattern
### 4.1 Architecture
- Each exchange runs on its own process 
  - Each crypto pair processes ticks on its own thread  
- _N_ times a second, a snapshot of the limit order book is taken, and 
persisted to a MongoDB

### 4.2 MongoDB Schema
  - The database schema consists of order book snapshots only:
    - `bids` and `asks`
      - `prices` = array of floats
      - `size` = array of floats (cumulative volume at price)
      - `count` = array of integers (total number of orders at a given price)
    - `upticks` and `downticks`
      - `size` = notional value of trades within the last period _N_ 
      (e.g., price x execution size)
      - `count` = total number of transactions that occured within the last 
      period _N_ (e.g., 5 trades)
    - `time` = `datetime.now()` of time the snapshot was taken

### 4.2 Limit Order Book
**SortedDict** python class is used for the limit order book
for the following reasons:
- **Price Insertions / deletions** within the limit order book
 can be performed with **O(log n)**
- **Getting / setting** values are performed with **O(1)**
- SortedDict interface is intuitive, thus making implementation easier

## To-dos:
1. Add aggregator to collapse GDAX and Bitfinex order books together