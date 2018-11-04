import pytz as tz


# ./crypto.py
SNAPSHOT_RATE = 15.0  # 0.25 = 4x second
BASKET = [('BTC-USD', 'tBTCUSD')]#,
          # ('ETH-USD', 'tETHUSD'),
          # ('LTC-USD', 'tLTCUSD'),
          # ('BCH-USD', 'tBCHUSD'),
          # ('ETC-USD', 'tETCUSD')]


# ./connector_components/client.py
COINBASE_ENDPOINT = 'wss://ws-feed.pro.coinbase.com'
BITFINEX_ENDPOINT = 'wss://api.bitfinex.com/ws/2'
MAX_RECONNECTION_ATTEMPTS = 300


# ./connector_components/book.py
MAX_BOOK_ROWS = 15


# ./connector_components/database.py
BATCH_SIZE = 100000
RECORD_DATA = False
MONGO_ENDPOINT = 'localhost'
ARCTIC_NAME = 'crypto.tickstore'


# ./database/database.py
TIMEZONE = tz.utc
