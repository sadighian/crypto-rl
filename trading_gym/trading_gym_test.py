import numpy as np
from datetime import datetime as dt
from trading_gym import TradingGym


if __name__ == '__main__':
    start_time = dt.now()

    lags = 0
    query = {
        'ccy': ['LTC-USD', 'tLTCUSD'],
        'start_date': 20181120,
        'end_date': 20181122
    }
    reward = 0.
    _env = TradingGym(query=query, lags=lags)

    for i in range(500):
        state, reward, done, info = _env.step(np.random.randint(5))
        print('observation: {}'.format(state[-2:]))
        if done:
            break

    print('Total reward: %f' % reward)
    elapsed = (dt.now() - start_time).seconds
    print('\nCompleted in %i seconds' % elapsed)
