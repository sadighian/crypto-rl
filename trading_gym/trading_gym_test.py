import numpy as np
import pandas as pd
from datetime import datetime as dt
from trading_gym import TradingGym
from simulator import Simulator as Sim
from sklearn.preprocessing import MinMaxScaler


if __name__ == '__main__':
    start_time = dt.now()

    # get testing data
    try:
        orderbook_data = pd.read_csv('~/Desktop/dev/crypto/trading_gym/orderbook_snapshot_history.csv')
        del orderbook_data[orderbook_data.columns[0]]
        orderbook_data['system_time'] = orderbook_data['system_time'].apply(lambda x: pd.to_datetime(x))
        orderbook_data['day'] = orderbook_data['system_time'].apply(lambda x: x.day)
        days = orderbook_data['day'].unique()

        sim = Sim()
        print('data loaded')
    except Exception as e:
        print('Exception: {}'.format(e))
        orderbook_data = None
        print('unable to load data')

    # split dataset into two: features fitting and normalized features
    fitting_data = orderbook_data.iloc[:orderbook_data.shape[0] // 2]
    fitting_data = fitting_data.dropna(axis=0)

    transforming_data = orderbook_data.iloc[orderbook_data.shape[0] // 2:]
    transforming_data = transforming_data.dropna(axis=0)

    data = fitting_data.drop(['system_time', 'day'], axis=1)
    scaler = MinMaxScaler()
    scaler.fit(fitting_data.drop(['system_time', 'day'], axis=1))

    data_ = pd.DataFrame(scaler.transform(data), columns=data.columns, index=data.index)

    params = {
        'data': data,
        'scaler': scaler,
        'training': True,
        'env_id': 'coinbase-bitfinex-v0',
        'step_size': 1,
        'fee': 0.006,
        'max_position': 1
    }
    total_reward = 0.0
    env = TradingGym(**params)

    for i in range(params['data'].shape[0]-2):
        if i % 2000 == 0:
            action = np.random.randint(3)
        else:
            action = 0
        state, reward, done, info = env.step(action)

        total_reward += reward

        if done:
            print('Done on %i step' % i)
            break

    print('Total reward: %f' % total_reward)
    elapsed = (dt.now() - start_time).seconds
    print('\nCompleted in %i seconds' % elapsed)
