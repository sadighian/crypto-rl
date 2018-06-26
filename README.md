# Multiprocessing Crypto Recorder
As of June 26th, 2018.
## 1. Purpose
Application designed to subscribe and record the
full limit order book data from **GDAX** and **Bitfinex** into a MongoDB 
for reinforcement learning simulations.

## 2. Scope
Application is intended to be used to record limit order book data for 
reinforcement learning modeling. Currently, there is no functionality 
developed to place order and actually trade.

## 3. Dependencies
- abc
- datetime
- pymongo
- time
- os
- json
- threading
- SortedDict
- websockets
- requests
- multiprocessing
- asyncio

## 4. Design Pattern
### 4.1 Architecture
- Both exchanges run on the same process
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

### 4.2 Limit Order OrderBook
**SortedDict** python class is used for the limit order book
for the following reasons:
- Price **Insertions** within the limit order book
 can be performed with **O(log n)**
- Price **Deletions** within the limit order book can be performed with **O(1)**
- **Getting / setting** values are performed with **O(1)**
- **SortedDict** interface is intuitive, thus making implementation easier

## 5. Appendix
### 5.1 Assumptions
- You know how to start up a mongo database and have mongoDb installed already
- You know how to clone projections using Git
- You know how to use a CLI

### 5.2 To-dos:
1. Create a back testing simulation environment using GYM
2. Integrate Tensorflow into trading model