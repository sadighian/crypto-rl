import logging
import os

import pytz as tz

# singleton for logging
logging.basicConfig(level=logging.INFO, format='[%(asctime)s] %(message)s')
LOGGER = logging.getLogger('crypto_rl_log')

# ./recorder.py
SNAPSHOT_RATE = 1.0  # For example, 0.25 = 4x per second
BASKET = [('BTC-USD', 'tBTCUSD'),
          # ('ETH-USD', 'tETHUSD'),
          # ('LTC-USD', 'tLTCUSD')
          ]

# ./data_recorder/connector_components/client.py
COINBASE_ENDPOINT = 'wss://ws-feed.pro.coinbase.com'
COINBASE_BOOK_ENDPOINT = 'https://api.pro.coinbase.com/products/%s/book'
BITFINEX_ENDPOINT = 'wss://api.bitfinex.com/ws/2'
MAX_RECONNECTION_ATTEMPTS = 100

# ./data_recorder/connector_components/book.py
MAX_BOOK_ROWS = 15
INCLUDE_ORDERFLOW = True

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
LIMIT_ORDER_FEE = 0.0
SLIPPAGE = 0.0005

# ./indicators/*
INDICATOR_WINDOW = [60 * i for i in [5, 15]]  # Convert minutes to seconds
INDICATOR_WINDOW_MAX = max(INDICATOR_WINDOW)
INDICATOR_WINDOW_FEATURES = [f'_{i}' for i in [5, 15]]  # Create labels
EMA_ALPHA = 0.99  # [0.9, 0.99, 0.999, 0.9999]

# agent penalty configs
ENCOURAGEMENT = 0.000000000001

# Data Directory
ROOT_PATH = os.path.dirname(os.path.realpath(__file__))
DATA_PATH = os.path.join(ROOT_PATH, 'data_recorder', 'database', 'data_exports')
