# ./main.py
MONGO_ENDPOINT = 'mongodb://localhost:27017/'
RECORD_DATA = True
SNAPSHOT_RATE = 0.25  # 0.25 = 4x second
# BASKET = [['BTC-USD', 'BCH-USD', 'ETH-USD', 'LTC-USD', 'BTC-EUR', 'ETH-EUR', 'BTC-GBP'],  # GDAX pairs
#           ['tBTCUSD', 'tBCHUSD', 'tETHUSD', 'tLTCUSD', 'tBTCEUR', 'tETHEUR', 'tBTCGBP']]  # Bitfinex pairs
BASKET = [['BTC-USD', 'ETH-USD', 'LTC-USD', 'BCH-USD'],
          ['tBTCUSD', 'tETHUSD', 'tLTCUSD', 'tBCHUSD']]


# ./common_components/client.py
GDAX_ENDPOINT = 'wss://ws-feed.pro.coinbase.com'
BITFINEX_ENDPOINT = 'wss://api.bitfinex.com/ws/2'
MAX_RECONNECTION_ATTEMPTS = 300


# ./common_components/book.py
MAX_BOOK_ROWS = 250