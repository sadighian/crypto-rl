# main.py
MONGO_ENDPOINT = 'mongodb://localhost:27017/'
RECORD_DATA = False
SNAPSHOT_RATE = 2.0  # 0.2 = 5x second

# common_components/client.py
GDAX_ENDPOINT = 'wss://ws-feed.pro.coinbase.com'
BITFINEX_ENDPOINT = 'wss://api.bitfinex.com/ws/2'
MAX_RECONNECTION_ATTEMPTS = 300
