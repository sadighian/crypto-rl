from simulator import Simulator
from coinbase_connector.coinbase_orderbook import CoinbaseOrderBook
from bitfinex_connector.bitfinex_orderbook import BitfinexOrderBook
from configurations.configs import TIMEZONE
from datetime import datetime as dt


query = {
    'ccy': ['BTC-USD', 'tBTCUSD'],
    'start_date': 20181120,
    'end_date': 20181121
}


def test_get_orderbook_snapshot_history(query):
    start_time = dt.now(TIMEZONE)

    coinbaseOrderBook = CoinbaseOrderBook(query['ccy'][0])
    bitfinexOrderBook = BitfinexOrderBook(query['ccy'][1])
    sim = Simulator()
    tick_history = sim.get_tick_history(query)

    if tick_history is None:
        print('Exiting due to no data being available.')
        return

    orderbook_snapshot_history = sim.get_orderbook_snapshot_history(coinbaseOrderBook,
                                                                    bitfinexOrderBook,
                                                                    tick_history)

    # Export to CSV to verify if order book reconstruction is accurate/good
    # NOTE: this is only to show that the functionality works and
    #       should be fed into an Environment for reinforcement learning.
    filename = './data_exports/{}_{}.csv'.format(query['ccy'][0], query['start_date'])
    sim.export_to_csv(data=orderbook_snapshot_history, filename=filename)

    elapsed = (dt.now(TIMEZONE) - start_time).seconds
    print('Completed %s in %i seconds' % (__name__, elapsed))
    print('DONE. EXITING %s' % __name__)


def test_query_env_states(query):
    start_time = dt.now(TIMEZONE)

    sim = Simulator()
    env_states = sim.query_env_states(query)

    sim.export_to_csv(data=env_states, filename='./env_states.csv')

    elapsed = (dt.now(TIMEZONE) - start_time).seconds
    print('Completed %s in %i seconds' % (__name__, elapsed))
    print('DONE. EXITING %s' % __name__)


def test_load_env_states():
    start_time = dt.now(TIMEZONE)

    sim = Simulator()
    fitting_filepath = './fitting_data.csv'
    env_filepath = './env_data.csv'
    env_states = sim.load_env_states(fitting_filepath=fitting_filepath, env_filepath=env_filepath)

    sim.export_to_csv(data=env_states, filename='./env_states.csv')

    elapsed = (dt.now(TIMEZONE) - start_time).seconds
    print('Completed %s in %i seconds' % (__name__, elapsed))
    print('DONE. EXITING %s' % __name__)


if __name__ == '__main__':
    """
    Entry point of simulation application
    """
    # test_get_orderbook_snapshot_history(query)
    query['start_date'] += 1
    query['end_date'] += 1
    test_get_orderbook_snapshot_history(query)
    # test_query_env_states(query)
    # test_load_env_states()
    # pass
