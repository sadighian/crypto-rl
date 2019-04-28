# Configurations
As of April 28, 2019.

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

### 2.7 BATCH_SIZE 
(`int`) Number of orders to append to a list before performing a batch insert
into the `Arctic Tick Store`

### 2.8 RECORD_DATA 
(`bool`) Flag for recording tick data (TRUE = Recording / FALSE = Not recording)

**Note** For simulations, this flag must be set to FALSE.

### 2.9 MONGO_ENDPOINT 
(`string`) Server address of the `Arctic Tick Store`

### 2.10 ARCTIC_NAME 
(`string`) Collection name for the `Arctic Tick Store`

### 2.11 TIMEZONE
(`pytz.tz`) Timezone for making `datetime.now()` calls (e.g., UTC)

### 2.12 BROKER_FEE
(`float`) Percentage of trade notional value that broker charges
as a fee
