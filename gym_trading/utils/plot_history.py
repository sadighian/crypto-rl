import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

SMALL_SIZE = 12
MEDIUM_SIZE = 14
BIGGER_SIZE = 16
plt.rc('font', size=SMALL_SIZE)  # controls default text sizes
plt.rc('axes', titlesize=SMALL_SIZE)  # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)  # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)  # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)  # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)  # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title


class Visualize(object):

    def __init__(self, columns: list or None, store_historical_observations: bool = True):
        """

        :param columns: List containing string column names for the data being stored.
        :param store_historical_observations: if TRUE, store observations, else disable
        """
        self._data = list()
        self._columns = columns

        # observation space for rendering
        self._store_historical_observations = store_historical_observations
        self._historical_observations = list()

    def add_observation(self, obs: np.ndarray) -> None:
        """

        :param obs: observation from the environment
        """
        if self._store_historical_observations:
            self._historical_observations.append(obs)

    def add(self, *args) -> None:
        """
        Add time step to visualizer.

        :param args: midpoint, buy trades, sell trades
        """
        self._data.append((args))

    def to_df(self) -> pd.DataFrame:
        """
        Get data in the form of a DataFrame.

        :return: Data captured throughout the episode
        """
        return pd.DataFrame(data=self._data, columns=self._columns)

    def reset(self) -> None:
        """
        Reset data for new episode.
        """
        self._data.clear()
        self._historical_observations.clear()

    def plot(self, history: pd.DataFrame or None = None, save_filename: str or None = None):
        """
        Plot episode performance:
            1.) Midpoint prices
            2.) Position count
            3.) Realized PnL

        :param history: data from past episode
        :param save_filename: Filename to save image as
        :return:
        """
        if isinstance(history, pd.DataFrame):
            data = history
        else:
            data = self.to_df()

        midpoints = data['midpoint'].values
        long_fills = data.loc[data['buys'] > 0., 'buys'].index.values
        short_fills = data.loc[data['sells'] > 0., 'sells'].index.values
        inventory = data['inventory'].values
        pnl = data['realized_pnl'].values

        fig, axs = plt.subplots(nrows=3, ncols=1, sharex=True, figsize=(14, 10))

        axs[0].plot(midpoints, label='midpoints', color='blue', alpha=0.6)
        axs[0].set_ylabel('Midpoint Price (USD)', color='black')
        # axs[0].set_xlabel('Number of steps (1 second each step)', color='black')

        # redundant labeling for all computer compatibility
        axs[0].set_facecolor("w")
        axs[0].tick_params(axis='x', colors='black')
        axs[0].tick_params(axis='y', colors='black')
        axs[0].spines['top'].set_visible(True)
        axs[0].spines['right'].set_visible(True)
        axs[0].spines['bottom'].set_visible(True)
        axs[0].spines['left'].set_visible(True)
        axs[0].spines['top'].set_color("black")
        axs[0].spines['right'].set_color("black")
        axs[0].spines['bottom'].set_color("black")
        axs[0].spines['left'].set_color("black")
        axs[0].grid(color='grey', linestyle='-', linewidth=0.25, alpha=0.5)
        for fill in long_fills:
            axs[0].scatter(x=fill, y=midpoints[fill], label='buys', alpha=0.9, color='green')

        for fill in short_fills:
            axs[0].scatter(x=fill, y=midpoints[fill], label='sells', alpha=0.9, color='red')

        axs[1].plot(inventory, label='inventory', color='orange')
        axs[1].axhline(0., color='grey')
        axs[1].set_ylabel('Inventory Size', color='black')
        # axs[1].set_xlabel('Number of steps (1 second each step)', color='black')
        # redundant labeling for all computer compatibility
        axs[1].set_facecolor("w")
        axs[1].tick_params(axis='x', colors='black')
        axs[1].tick_params(axis='y', colors='black')
        axs[1].spines['top'].set_visible(True)
        axs[1].spines['right'].set_visible(True)
        axs[1].spines['bottom'].set_visible(True)
        axs[1].spines['left'].set_visible(True)
        axs[1].spines['top'].set_color("black")
        axs[1].spines['right'].set_color("black")
        axs[1].spines['bottom'].set_color("black")
        axs[1].spines['left'].set_color("black")
        axs[1].grid(color='grey', linestyle='-', linewidth=0.25, alpha=0.5)

        axs[2].plot(pnl, label='Realized PnL', color='purple')
        axs[2].axhline(0., color='grey')
        axs[2].set_ylabel("PnL (%)", color='black')
        axs[2].set_xlabel('Number of steps (1 second each step)', color='black')
        # redundant labeling for all computer compatibility
        axs[2].set_facecolor("w")
        axs[2].tick_params(axis='x', colors='black')
        axs[2].tick_params(axis='y', colors='black')
        axs[2].spines['top'].set_visible(True)
        axs[2].spines['right'].set_visible(True)
        axs[2].spines['bottom'].set_visible(True)
        axs[2].spines['left'].set_visible(True)
        axs[2].spines['top'].set_color("black")
        axs[2].spines['right'].set_color("black")
        axs[2].spines['bottom'].set_color("black")
        axs[2].spines['left'].set_color("black")
        axs[2].grid(color='grey', linestyle='-', linewidth=0.25, alpha=0.5)

        if save_filename is None:
            plt.show()
        else:
            plt.savefig("{}.png".format(save_filename))

    def plot_obs(self) -> None:
        """
        Plot observations from an entire episode as one image.
        """
        fig, ax = plt.subplots(figsize=(16, 10))
        observations = np.asarray(self._historical_observations, dtype=np.float32)
        im = ax.imshow(observations,
                       interpolation=None,
                       cmap=cm.seismic,
                       origin='lower',
                       aspect='auto',
                       vmax=observations.max(),
                       vmin=observations.min())

        plt.show()
