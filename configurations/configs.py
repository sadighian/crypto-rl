import pytz as tz


# ./recorder.py
SNAPSHOT_RATE = 10  # 0.25 = 4x second
BASKET = [('BTC-USD', 'tBTCUSD'),
          ('ETH-USD', 'tETHUSD'),
          ('LTC-USD', 'tLTCUSD'),
          ('BCH-USD', 'tBCHUSD')]


# ./data_recorder/connector_components/client.py
COINBASE_ENDPOINT = 'wss://ws-feed.pro.coinbase.com'
BITFINEX_ENDPOINT = 'wss://api.bitfinex.com/ws/2'
MAX_RECONNECTION_ATTEMPTS = 300


# ./data_recorder/connector_components/book.py
MAX_BOOK_ROWS = 15
INCLUDE_ORDERFLOW = True
BOOK_DIGITS = 2  # Used to aggregate prices
AGGREGATE = False


# ./data_recorder/database/database.py
BATCH_SIZE = 100000
RECORD_DATA = False
MONGO_ENDPOINT = 'localhost'
ARCTIC_NAME = 'crypto.tickstore'
TIMEZONE = tz.utc


# ./data_recorder/database/simulator.py
SNAPSHOT_RATE_IN_MICROSECONDS = 1000000  # 1 second


# ./gym_trading/utils/broker.py
MARKET_ORDER_FEE = 0.0020
LIMIT_ORDER_FEE = 0.0010


# ./indicators/*
window_intervals = [5, 15, 30]
INDICATOR_WINDOW = [60*i for i in window_intervals]
INDICATOR_WINDOW_MAX = max(INDICATOR_WINDOW)
INDICATOR_WINDOW_FEATURES = ['_{}'.format(i) for i in window_intervals]
EMA_ALPHA = 0.99  # [0.9, 0.99, 0.999, 0.9999]
