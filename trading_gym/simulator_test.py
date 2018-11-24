from simulator import Simulator
from coinbase_connector.coinbase_orderbook import CoinbaseOrderBook
from bitfinex_connector.bitfinex_orderbook import BitfinexOrderBook
from configurations.configs import TIMEZONE
from datetime import datetime as dt


query = {
    'ccy': ['LTC-USD', 'tLTCUSD'],
    'start_date': 20181120,
    'end_date': 20181122
}
lags = 1


def test_get_orderbook_snapshot_history(query):
    start_time = dt.now(TIMEZONE)

    coinbaseOrderBook = CoinbaseOrderBook(query['ccy'][0])
    bitfinexOrderBook = BitfinexOrderBook(query['ccy'][1])
    sim = Simulator()
    tick_history = sim.get_tick_history(query)

    if tick_history is None:
        print('Exiting due to no data being available.')
        return

    orderbook_snapshot_history = sim.get_orderbook_snapshot_history(coinbaseOrderBook, bitfinexOrderBook, tick_history)

    # Export to CSV to verify if order book reconstruction is accurate/good
    # NOTE: this is only to show that the functionality works and
    #       should be fed into an Environment for reinforcement learning.
    sim.export_snapshots_to_csv(sim.get_feature_labels(True), orderbook_snapshot_history)

    elapsed = (dt.now(TIMEZONE) - start_time).seconds
    print('Completed %s in %i seconds' % (__name__, elapsed))
    print('DONE. EXITING %s' % __name__)


def test_get_env_data(query, lags):
    start_time = dt.now(TIMEZONE)

    sim = Simulator()
    env_data = sim.get_env_data(query,
                                CoinbaseOrderBook(query['ccy'][0]),
                                BitfinexOrderBook(query['ccy'][1]),
                                lags=lags)
    sim.export_env_data_to_csv(env_data)

    elapsed = (dt.now(TIMEZONE) - start_time).seconds
    print('Completed %s in %i seconds' % (__name__, elapsed))
    print('DONE. EXITING %s' % __name__)


if __name__ == '__main__':
    """
    Entry point of simulation application
    """
    test_get_orderbook_snapshot_history(query)
    # test_get_env_data(query, lags)
    pass
