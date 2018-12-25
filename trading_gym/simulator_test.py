from simulator import Simulator
from configurations.configs import TIMEZONE
from datetime import datetime as dt
import pandas as pd


query = {
    'ccy': ['ETH-USD', 'tETHUSD'],
    'start_date': 20181120,
    'end_date': 20181121
}


def test_get_tick_history():
    start_time = dt.now(TIMEZONE)

    sim = Simulator()
    tick_history = sim.get_tick_history(query=query)

    elapsed = (dt.now(TIMEZONE) - start_time).seconds
    print('Completed %s in %i seconds' % (__name__, elapsed))
    print('DONE. EXITING %s' % __name__)


def test_get_orderbook_snapshot_history():
    start_time = dt.now(TIMEZONE)

    sim = Simulator()
    orderbook_snapshot_history = sim.get_orderbook_snapshot_history(query=query)

    sim.export_to_csv(data=orderbook_snapshot_history, filename=query['ccy'][0], compress=True)

    elapsed = (dt.now(TIMEZONE) - start_time).seconds
    print('Completed %s in %i seconds' % (__name__, elapsed))
    print('DONE. EXITING %s' % __name__)


def test_extract_features():
    start_time = dt.now(TIMEZONE)

    sim = Simulator()
    sim.extract_features(query, num_of_days=2)

    elapsed = (dt.now(TIMEZONE) - start_time).seconds
    print('Completed %s in %i seconds' % (__name__, elapsed))
    print('DONE. EXITING %s' % __name__)


if __name__ == '__main__':
    """
    Entry point of simulation application
    """
    # test_get_tick_history()
    # test_get_orderbook_snapshot_history()
    test_extract_features()
