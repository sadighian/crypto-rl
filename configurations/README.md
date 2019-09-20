# Configurations
As of September 20, 2019.

## 1. Overview
The `configurations` module contains the settings implemented throughout the project.

## 2. Configuration Definitions
### 2.1 SNAPSHOT_RATE
(`float`) Number of times per second to take a snapshot of the order book
 (e.g., 0.25 will take a snapshot 4x a second)

### 2.2 BASKET
(`list` of `tuples`) Security names that the application will subscribe to.
The current syntax is [(coinbase security name, bitfinex security name), (), ...]

### 2.3 COINBASE_ENDPOINT
(`string`) Websocket endpoint for the tick data subscription

### 2.4 BITFINEX_ENDPOINT
(`string`) Websocket endpoint for the tick data subscription

### 2.5 MAX_RECONNECTION_ATTEMPTS
(`int`) Number of attempts the application can make to resubscribe to 
 the websocket channels, given a disconnection event

### 2.6 MAX_BOOK_ROWS 
(`int`) Number of rows in the order book to render during snapshots

### 2.7 INCLUDE_ORDERFLOW
(`bool`) Order arrival rates; if TRUE, include in feature set

### 2.8 BOOK_DIGITS
(`int`) Number of significant figures used for rounding LOB prices

### 2.9 AGGREGATE
(`bool`) If TRUE, round LOB prices by the specified `BOOK_DIGITS`

### 2.10 BATCH_SIZE 
(`int`) Number of orders to append to a list before performing a batch insert
into the `Arctic Tick Store`

### 2.11 RECORD_DATA 
(`bool`) Flag for recording tick data (TRUE = Recording / FALSE = Not recording)

**Note** For simulations, this flag must be set to FALSE.

### 2.12 MONGO_ENDPOINT 
(`string`) Server address of the `Arctic Tick Store`

### 2.13 ARCTIC_NAME 
(`string`) Collection name for the `Arctic Tick Store`

### 2.14 TIMEZONE
(`pytz.tz`) Timezone for making `datetime.now()` calls (e.g., UTC)

### 2.15 MARKET_ORDER_FEE
(`float`) Percentage of trade notional value that broker charges as a
fee for a market order

### 2.16 LIMIT_ORDER_FEE
(`float`) Percentage of trade notional value that broker charges as a
fee for a limit order

### 2.17 INDICATOR_WINDOW
(`list of int`) Rolling windows used by indicators (note: actual numbers
are calculated taking the $lob_snapshot_rate * 60_seconds_per_min *
num_of_mins = window_size$

### 2.18 INDICATOR_WINDOW_MAX
(`int`) The largest rolling window in `INDICATOR_WINDOW`

### 2.19 INDICATOR_WINDOW_FEATURES
(`list of str`) Feature label stubs to indentify the rolling window
lengths specified within `INDICATOR_WINDOW`

### 2.20 EMA_ALPHA
(`None`, `float`, or `list`) Flag to use Exponential Moving Average for
smoothing indicator and observation space values. If `None` is provide,
raw data is used; if a `float` is provided, the raw data values are
smoothed using the alpha number provided; if `list` is provided, the raw
data values are smoothed for each alpha in the list (i.e., if 3 alphas
are in the list, then the feature space dimension is multiplied by 3).