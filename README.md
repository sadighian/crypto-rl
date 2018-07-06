# Multiprocessing Crypto Recorder
As of July 6th, 2018.

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
- asyncio
- datetime
- json
- multiprocessing
- os
- pymongo
- requests
- SortedDict
- threading
- time
- websockets

## 4. Design Pattern
### 4.1 Architecture
- Each crypto pair (e.g., Bitcoin-USD) run on its own `Process`
  - Each exchange data feed is processed in its own `Thread` within the 
  parent crypto pair `Process`
- _N_ times a second, a snapshot of the limit order book is taken, and 
persisted to a MongoDB

![Design Pattern](assets/design-pattern.png)

### 4.2 MongoDB Schema
  - The database schema consists of order book snapshots only:
    - `gdax` or `bitfinex` (string) exchange name
        - `bids` and `asks` (string) order book side
          - `prices` = (array of floats) order price
          - `size` = array of floats (cumulative volume at price)
          - `count` = (array of integers) total number of 
          orders at a given price
        - `upticks` and `downticks` (string) buy or sell transactions
          - `size` = (float) notional value of trades within the 
          last period _N_ 
          (e.g., price x execution size)
          - `count` = (int) total number of transactions that occurred 
          within the last 
          period _N_ (e.g., 5 trades)
        - `time` = `datetime.now()` of time the snapshot was taken

### 4.3 Limit Order Book
**SortedDict** python class is used for the limit order book
for the following reasons:
- Sorted Price **Insertions** within the limit order book
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