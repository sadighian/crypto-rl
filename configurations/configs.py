import pytz as tz


# ./recorder.py
SNAPSHOT_RATE = 10  # 0.25 = 4x second
BASKET = [('BTC-USD', 'tBTCUSD'),
          ('ETH-USD', 'tETHUSD'),
          ('LTC-USD', 'tLTCUSD'),
          ('BCH-USD', 'tBCHUSD')]


# ./connector_components/client.py
COINBASE_ENDPOINT = 'wss://ws-feed.pro.coinbase.com'
BITFINEX_ENDPOINT = 'wss://api.bitfinex.com/ws/2'
MAX_RECONNECTION_ATTEMPTS = 300


# ./connector_components/book.py
MAX_BOOK_ROWS = 15


# ./database/database.py
BATCH_SIZE = 100000
RECORD_DATA = False
MONGO_ENDPOINT = 'localhost'
ARCTIC_NAME = 'crypto.tickstore'


# ./database/database.py
TIMEZONE = tz.utc


# ./simulator.py
BROKER_FEE = 0.003


# ./indicators/*
INDICATOR_WINDOW = [2*60*i for i in [5, 15, 30]]
INDICATOR_WINDOW_MAX = max(INDICATOR_WINDOW)
INDICATOR_WINDOW_FEATURES = ['_{}'.format(i) for i in [5, 15, 30]]
