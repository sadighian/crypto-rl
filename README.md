# Deep Reinforcement Learning Toolkit for Cryptocurrencies
As of March 04, 2019.

## 1. Purpose
The purpose of this application is to provide a toolkit to:
 - **Record** full limit order book and trade tick data from two 
 exchanges (**Coinbase Pro** and **Bitfinex**) into an Arctic 
 Tickstore database (i.e., MongoDB), 
 - **Replay** recorded historical data to derive feature sets for training
 - **Train** a Dueling Deep Q-Network (DDQN) agent to trade cryptocurrencies.

![Design Pattern](images/design-pattern-high-level.PNG)

## 2. Scope
Application is intended to be used to record and simulate limit order book data 
for reinforcement learning modeling. Currently, there is no functionality 
developed to place an order or automate trading.

## 3. Dependencies
- abc
- [arctic](https://github.com/manahl/arctic)
- asyncio
- datetime
- gym
- json
- keras
- keras-rl
- multiprocessing
- numpy
- os
- pandas
- pytz
- requests
- sklearn
- SortedDict
- tensorflow-gpu
- threading
- time
- websockets

## 4. Design Patterns

### 4.1 Data Recorder
(`./recorder.py`) The design pattern is intended to serve as a foundation for implementing a trading strategy.
#### 4.1.1 Recorder Architecture
- Each crypto pair (e.g., Bitcoin-USD) runs on its own `Process`
  - Each exchange data feed is processed in its own `Thread` within the parent crypto pair `Process`
  - A timer for periodic polling (or order book snapshots--see `mongo-integration` or `arctic-book-snapshot` 
  branch) runs on a separate thread

![Design Pattern](images/design-pattern.png)

#### 4.1.2 Tick Store Data Model
**Arctic tick store** is the database implementation of choice for this project for the 
following reasons:
 - ManAHL created and open sourced
 - Superior performance metrics (e.g., 10x data compression)

The **Arctic Tick Store** data model is essentially a `list` of `dict`s, where 
each `dict` is an incoming **tick** from the exchanges.
- Each `list` consists of `./configurations/BATCH_SIZE` ticks (e.g., 100,000 ticks)
- Per the Arctic Tick Store design, all currency pairs are stored in the **same** MongoDB collection

### 4.2 Limit Order Book Implementation
**SortedDict** pure python class is used for the limit order book
for the following reasons:
- Sorted Price **Insertions** within the limit order book
 can be performed with **O(log n)**
- Price **Deletions** within the limit order book can be performed with **O(log n)**
- **Getting / setting** values are performed with **O(1)**
- **SortedDict** interface is intuitive, thus making implementation easier

### 4.3 Reinforcement Learning Environment
(`./trading_gym/trading_gym.py`) Implementation of the `gym` signatures for a 
markov decision process (although our environment is POMDP).

(`./trading_gym/agent.py`) Implementation of a double-deuling q-network using a 
shallow network model. This class is where you can modify the network architecture when
performing network architecture research.

(`./experiment.py`) This class is the entrypoint for running simulations for training and 
evaluating trained DQN models.

## 5. Examples and Usage
### 5.1 Recorder.py
Class for recording limit order and trade data. 

**Step 1:**
Go to the configurations.configs.py and define the crypto currencies
you would like to subscribe and record. Note: the first column of CCYs are 
Coinbase Pro currency names, and the second column of CCYs are Bitfinex's.
```
SNAPSHOT_RATE = 15.  # 0.25 = 4x second
BASKET = [('BTC-USD', 'tBTCUSD'),
         ('ETH-USD', 'tETHUSD'),
         ('LTC-USD', 'tLTCUSD'),
         ('BCH-USD', 'tBCHUSD'),
         ('ETC-USD', 'tETCUSD')]
RECORD_DATA = True
```

**Step 2:**
Open a CLI and start recording full limit order book and trade data.
 ```
 python3 recorder.py
 ```

### 5.2 Experiment.py
Class for running experiments. 

**Step 1:**
Record streaming data using `./recorder.py` (see above)

**Step 2:**
Create derived feature sets and export to `csv` or `xz`
```
# test case for exporting data sets to csv or xz
python3 simulator_test.test_extract_features()
```

**Step 3:**
Open a CLI and run an experiment:
```
python3 experiment.py
```

Note: the branch is currently configured to load the historical data from a xz
compressed csv (opposed to generating data on-demand using the Arctic Tick 
Store to save time). To generate the xz csv, run the 
`Simulator.extract_features()` method (assuming you've recorded data already!).
Refer to `./trading_gym/simulator_test.py` for an example.

## 6. Appendix
### 6.1 Branches
There are multiple branches of this project, each with a different implementation pattern for persisting data:
 - **FULL** branch is intended to be the foundation for a fully automated trading system (i.e., implementation of
 design patterns that are ideal for a trading system that requires parallel processing) and  persists streaming 
 tick data into an **Arctic Tick Store**
 
 **Note:** the branches below (i.e., lightweight, order book snapshot, mongo integration) are no longer actively maintained as of October 2018, 
 and are here for reference.
 - **LIGHT WEIGHT** branch is intended to record streaming data more efficiently than the __full__ branch (i.e., 
 all websocket connections are made from a single process __and__ the limit order book is not maintained) and
 persists streaming tick data into an **Arctic tick store**
 - **ORDER BOOK SNAPSHOT** branch has the same design pattern as the __full__ branch, but instead of recording 
 streaming ticks, snapshots of the limit order book are taken every **N** seconds and persisted 
 into an **Arctic tick store**
 - **MONGO INTEGRATION** branch is the same implementation as **ORDER BOOK SNAPSHOT**, with the difference being 
 a standard MongoDB is used, rather than Arctic. This branch was originally used to benchmark Arctic's 
 performance and is not up to date with the **FULL** branch.

### 6.2 Assumptions
- You know how to start up a MongoDB database and have mongoDB installed already
- You know how to use Git tools
- You are familiar with python3

### 6.3 To-dos:
1. Create DockerFile so that simulations can be rapidly deployed in the cloud (e.g., AWS Fargate)
