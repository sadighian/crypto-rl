# Data Recorder
As of April 28th, 2019.

(`recorder.py`) The design pattern is intended to serve as a 
foundation for implementing a trading strategy.


## 1. Recorder Architecture
- Each crypto pair (e.g., Bitcoin-USD) runs on its own `Process`
  - Each exchange data feed is processed in its own `Thread` within the 
  parent crypto pair `Process`
  - A timer for periodic polling (or order book snapshots--see 
  `mongo-integration` or `arctic-book-snapshot` branch) runs on 
  a separate thread

![plot_order_arrivals](../design_patterns/design-pattern.png)

## 2. Tick Store Data Model
**Arctic tick store** is the database implementation of choice for 
this project for the 
following reasons:
 - ManAHL created and open sourced
 - Superior performance metrics (e.g., 10x data compression)

The **Arctic Tick Store** data model is essentially a `list` of `dict`s, where 
each `dict` is an incoming **tick** from the exchanges.
- Each `list` consists of `configurations/configs.BATCH_SIZE` ticks 
(e.g., 100,000 ticks)
- Per the Arctic Tick Store design, all currency pairs are stored 
in the **same** MongoDB collection

## 3. Limit Order Book Implementation
**SortedDict** pure python class is used for the limit order book
for the following reasons:
- Sorted Price **Insertions** within the limit order book
 can be performed with **O(log n)**
- Price **Deletions** within the limit order book can be performed 
with **O(log n)**
- **Getting / setting** values are performed with **O(1)**
- **SortedDict** interface is intuitive, thus making implementation easier

![plot_order_arrivals](../design_patterns/plot_order_arrivals.png)