import os
from datetime import datetime as dt

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from configurations import DATA_PATH, EMA_ALPHA, LOGGER, MAX_BOOK_ROWS, TIMEZONE
from indicators import apply_ema_all_data, load_ema, reset_ema


class DataPipeline(object):

    def __init__(self, alpha: float or list or None = EMA_ALPHA):
        """
        Data Pipeline constructor.
        """
        self.alpha = alpha
        self.ema = load_ema(alpha=alpha)
        self._scaler = StandardScaler()

    def reset(self) -> None:
        """
        Reset data pipeline.
        """
        self._scaler = StandardScaler()
        self.ema = reset_ema(ema=self.ema)

    @staticmethod
    def import_csv(filename: str) -> pd.DataFrame:
        """
        Import an historical tick file created from the export_to_csv() function.

        :param filename: Full file path including filename
        :return: (panda.DataFrame) historical limit order book data
        """
        start_time = dt.now(tz=TIMEZONE)

        if 'xz' in filename:
            data = pd.read_csv(filepath_or_buffer=filename, index_col=0,
                               compression='xz', engine='c')
        elif 'csv' in filename:
            data = pd.read_csv(filepath_or_buffer=filename, index_col=0, engine='c')
        else:
            LOGGER.warn('Error: file must be a csv or xz')
            data = None

        elapsed = (dt.now(tz=TIMEZONE) - start_time).seconds
        LOGGER.info('Imported %s from a csv in %i seconds' % (filename[-25:], elapsed))
        return data

    def fit_scaler(self, orderbook_snapshot_history: pd.DataFrame) -> None:
        """
        Scale limit order book data for the neural network.

        :param orderbook_snapshot_history: Limit order book data
                from the previous day
        :return: (void)
        """
        self._scaler.fit(orderbook_snapshot_history)

    def scale_data(self, data: pd.DataFrame) -> np.ndarray:
        """
        Standardize data.

        :param data: (np.array) all data in environment
        :return: (np.array) normalized observation space
        """
        return self._scaler.transform(data)

    @staticmethod
    def _midpoint_diff(data: pd.DataFrame) -> pd.DataFrame:
        """
        Take log difference of midpoint prices
                log(price t) - log(price t-1)

        :param data: (pd.DataFrame) raw data from LOB snapshots
        :return: (pd.DataFrame) with midpoint prices normalized
        """
        data['midpoint'] = np.log(data['midpoint'].values)
        data['midpoint'] = (data['midpoint'] - data['midpoint'].shift(1)
                            ).fillna(method='bfill')
        return data

    @staticmethod
    def _decompose_order_flow_information(data: pd.DataFrame) -> pd.DataFrame:
        """
        Transform raw [market, limit, cancel] notional values into a single OFI.

        :param data: snapshot data imported from `self.import_csv`
        :return: LOB data with OFI
        """
        # Derive column names for filtering OFI data
        event_columns = dict()
        for event_type in ['market_notional', 'limit_notional', 'cancel_notional']:
            event_columns[event_type] = [col for col in data.columns.tolist() if
                                         event_type in col]

        # Derive the number of rows that have been rendered in the LOB
        number_of_levels = len(event_columns['market_notional']) // 2

        # Calculate OFI = LIMIT - MARKET - CANCEL
        ofi_data = data[event_columns['limit_notional']].values - \
                   data[event_columns['market_notional']].values - \
                   data[event_columns['cancel_notional']].values

        # Convert numpy to DataFrame
        ofi_data = pd.DataFrame(data=ofi_data,
                                columns=[f'ofi_bid_{i}' for i in range(number_of_levels)] +
                                        [f'ofi_ask_{i}' for i in range(number_of_levels)],
                                index=data.index)

        # Merge with original data set
        data = pd.concat((data, ofi_data), axis=1)

        # Drop MARKET, LIMIT, and CANCEL columns from original data set
        for event_type in {'market_notional', 'limit_notional', 'cancel_notional'}:
            data = data.drop(event_columns[event_type], axis=1)

        return data

    @staticmethod
    def get_imbalance_labels() -> list:
        """
        Get a list of column labels for notional order imbalances.
        """
        imbalance_labels = [f'notional_imbalance_{row}' for row in range(MAX_BOOK_ROWS)]
        imbalance_labels += ['notional_imbalance_mean']
        return imbalance_labels

    @staticmethod
    def _get_notional_imbalance(data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate order imbalances per price level, their mean & standard deviation.

        Order Imbalances are calculated by:
            = (bid_quantity - ask_quantity) / (bid_quantity + ask_quantity)

        ...thus scale from [-1, 1].

        :param data: raw/un-normalized LOB snapshot data
        :return: (pd.DataFrame) order imbalances at N-levels, the mean & std imbalance
        """
        # Create the column names for making a data frame (also used for debugging)
        bid_notional_columns, ask_notional_columns, imbalance_columns = [], [], []
        for i in range(MAX_BOOK_ROWS):
            bid_notional_columns.append(f'bids_notional_{i}')
            ask_notional_columns.append(f'asks_notional_{i}')
            imbalance_columns.append(f'notional_imbalance_{i}')
        # Acquire bid and ask notional data
        # Reverse the bids to ascending order, so that they align with the asks
        bid_notional = data[bid_notional_columns].to_numpy(dtype=np.float32)  # [::-1]
        ask_notional = data[ask_notional_columns].to_numpy(dtype=np.float32)

        # Transform to cumulative imbalances
        bid_notional = np.cumsum(bid_notional, axis=1)
        ask_notional = np.cumsum(ask_notional, axis=1)

        # Calculate the order imbalance
        imbalances = ((bid_notional - ask_notional) + 1e-5) / \
                     ((bid_notional + ask_notional) + 1e-5)
        imbalances = pd.DataFrame(imbalances, columns=imbalance_columns,
                                  index=data.index).fillna(0.)
        # Add meta data to features (mean)
        imbalances['notional_imbalance_mean'] = imbalances[imbalance_columns].mean(axis=1)
        return imbalances

    def load_environment_data(self, fitting_file: str, testing_file: str,
                              include_imbalances: bool = True, as_pandas: bool = False) \
            -> (pd.DataFrame, pd.DataFrame, pd.DataFrame):
        """
        Import and scale environment data set with prior day's data.

        Midpoint gets log-normalized:
            log(price t) - log(price t-1)

        :param fitting_file: prior trading day
        :param testing_file: current trading day
        :param include_imbalances: if TRUE, include LOB imbalances
        :param as_pandas: if TRUE, return data as DataFrame, otherwise np.array
        :return: (pd.DataFrame or np.array) scaled environment data
        """
        # Import data used to fit scaler
        fitting_data_filepath = os.path.join(DATA_PATH, fitting_file)
        fitting_data = self.import_csv(filename=fitting_data_filepath)

        # Derive OFI statistics
        fitting_data = self._decompose_order_flow_information(data=fitting_data)
        # Take the log difference of midpoint prices
        fitting_data = self._midpoint_diff(data=fitting_data)  # normalize midpoint
        # If applicable, smooth data set with EMA(s)
        fitting_data = apply_ema_all_data(ema=self.ema, data=fitting_data)
        # Fit the scaler
        self.fit_scaler(fitting_data)
        # Delete data from memory
        del fitting_data

        # Import data to normalize and use in environment
        data_used_in_environment = os.path.join(DATA_PATH, testing_file)
        data = self.import_csv(filename=data_used_in_environment)

        # Raw midpoint prices for back-testing environment
        midpoint_prices = data['midpoint']

        # Copy of raw LOB snapshots for normalization
        normalized_data = self._midpoint_diff(data.copy(deep=True))

        # Preserve the raw data and drop unnecessary columns
        data = data.drop([col for col in data.columns.tolist()
                          if col in ['market', 'limit', 'cancel']], axis=1)

        # Derive OFI statistics
        normalized_data = self._decompose_order_flow_information(data=normalized_data)

        normalized_data = apply_ema_all_data(ema=self.ema, data=normalized_data)

        # Get column names for putting the numpy values into a data frame
        column_names = normalized_data.columns.tolist()
        # Scale data with fitting data set
        normalized_data = self.scale_data(normalized_data)
        # Remove outliers
        normalized_data = np.clip(normalized_data, -10., 10.)
        # Put data in a data frame
        normalized_data = pd.DataFrame(normalized_data,
                                       columns=column_names,
                                       index=midpoint_prices.index)

        if include_imbalances:
            LOGGER.info('Adding order imbalances...')
            # Note: since order imbalance data is scaled [-1, 1], we do not apply
            # z-score to the imbalance data
            imbalance_data = self._get_notional_imbalance(data=data)
            self.ema = reset_ema(self.ema)
            imbalance_data = apply_ema_all_data(ema=self.ema, data=imbalance_data)
            normalized_data = pd.concat((normalized_data, imbalance_data), axis=1)

        if as_pandas is False:
            midpoint_prices = midpoint_prices.to_numpy(dtype=np.float64)
            data = data.to_numpy(dtype=np.float32)
            normalized_data = normalized_data.to_numpy(dtype=np.float32)

        return midpoint_prices, data, normalized_data
