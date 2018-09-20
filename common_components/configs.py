# ./main.py
SNAPSHOT_RATE = 0.25  #  0.25 = 4x second
BASKET = [('BTC-USD', 'tBTCUSD')] #,
          # ('ETH-USD', 'tETHUSD'),
          # ('LTC-USD', 'tLTCUSD'),
          # ('BCH-USD', 'tBCHUSD'),
          # ('ETC-USD', 'tETCUSD'),
          # ('BTC-GBP', 'tBTCGBP'),
          # ('BTC-EUR', 'tBTCEUR')]


# ./common_components/client.py
GDAX_ENDPOINT = 'wss://ws-feed.pro.coinbase.com'
BITFINEX_ENDPOINT = 'wss://api.bitfinex.com/ws/2'
MAX_RECONNECTION_ATTEMPTS = 300


# ./common_components/book.py
MAX_BOOK_ROWS = 250


# ./common_components/database.py
BATCH_SIZE = 100
RECORD_DATA = True
MONGO_ENDPOINT = 'localhost'
ARCTIC_NAME = 'crypto.tickstore'
