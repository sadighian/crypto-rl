# Multiprocessing Crypto Recorder (Lightewight - streaming ticks)
As of August 17th, 2018.

## 1. Purpose
Application is designed to subscribe and record 
full limit order book data from **GDAX** and **Bitfinex** into an Arctic Tickstore 
database (i.e., MongoDB) to perform reinforcement learning research.

## 2. Scope
Application is intended to be used to record limit order book data for 
reinforcement learning modeling. Currently, there is no functionality 
developed to place order and actually trade.

## 3. Dependencies
- abc
- [arctic](https://github.com/manahl/arctic)
- asyncio
- datetime
- json
- multiprocessing
- os
- pandas
- requests
- threading
- time
- websockets

## 4. Design Pattern
### 4.1 Architecture
- Each crypto pair for each exchange (e.g., Bitcoin-USD) runs on its own `Thread`

### 4.2 Arctic Schema
**Arctic tick store** is the database implementation for this project for the 
following reasons:
 - Open sourced reliability
 - Superior performance metrics (e.g., 10x data compression)

The **arctic tick store** data model is essentially a `list` of `dict`s, where 
each `dict` is an incoming tick from the exchanges.
- Each `list` consists of `./common_components/configs.CHUNK_SIZE` ticks (e.g., 100,000)
- All currency pairs are stored in the **same** MongoDB collection

### 4.3 Limit Order Book
This is the `lightweight` branch, which does not have a limit order book.

## 5. Appendix
### 5.1 Assumptions
- You know how to start up a Mongo database and have mongoDb installed already
- You know how to use Git tools
- You know how to use a CLI

### 5.2 To-dos:
1. Create a back testing simulation environment using GYM
2. Integrate Tensorflow into trading model
