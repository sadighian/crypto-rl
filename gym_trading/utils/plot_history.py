import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

SMALL_SIZE = 12
MEDIUM_SIZE = 14
BIGGER_SIZE = 16
plt.rc('font', size=SMALL_SIZE)  # controls default text sizes
plt.rc('axes', titlesize=SMALL_SIZE)  # font size of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)  # font size of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)  # font size of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)  # font size of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)  # legend font size
plt.rc('figure', titlesize=BIGGER_SIZE)  # font size of the figure title


def plot_observation_space(observation: np.ndarray,
                           labels: str,
                           save_filename: str or None = None) -> None:
    """
    Represent all the observation spaces seen by the agent as one image.
    """
    fig, ax = plt.subplots(figsize=(16, 10))
    im = ax.imshow(observation,
                   interpolation='none',
                   cmap=cm.get_cmap('seismic'),
                   origin='lower',
                   aspect='auto',
                   vmax=observation.max(),
                   vmin=observation.min())
    plt.xticks(range(len(labels)), labels, rotation='vertical')
    plt.tight_layout()

    if save_filename is None:
        plt.show()
    else:
        plt.savefig(f"{save_filename}_OBS.png")
        plt.close(fig)


class Visualize(object):

    def __init__(self,
                 columns: list or None,
                 store_historical_observations: bool = True):
        """
        Helper class to store episode performance.

        :param columns: Column names (or labels) for rending data
        :param store_historical_observations: if TRUE, store observation
            space for rendering as an image at the end of an episode
        """
        self._data = list()
        self._columns = columns

        # Observation space for rendering
        self._store_historical_observations = store_historical_observations
        self._historical_observations = list()
        self.observation_labels = None

    def add_observation(self, obs: np.ndarray) -> None:
        """
        Append current time step of observation to list for rendering
        observation space at the end of an episode.

        :param obs: Current time step observation from the environment
        """
        if self._store_historical_observations:
            self._historical_observations.append(obs)

    def add(self, *args):
        """
        Add time step to visualizer.

        :param args: midpoint, buy trades, sell trades
        :return:
        """
        self._data.append(args)

    def to_df(self) -> pd.DataFrame:
        """
        Get episode history of prices and agent transactions in the form of a DataFrame.

        :return: DataFrame with episode history of prices and agent transactions
        """
        return pd.DataFrame(data=self._data, columns=self._columns)

    def reset(self) -> None:
        """
        Reset data for new episode.
        """
        self._data.clear()
        self._historical_observations.clear()

    def plot_episode_history(self, history: pd.DataFrame or None = None,
                             save_filename: str or None = None) -> None:
        """
        Plot this entire history of an episode including:
            1) Midpoint prices with trade executions
            2) Inventory count at every step
            3) Realized PnL at every step

        :param history: data from past episode
        :param save_filename: Filename to save image as
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

        heights = [6, 2, 2]
        widths = [14]
        gs_kw = dict(width_ratios=widths, height_ratios=heights)
        fig, axs = plt.subplots(nrows=len(heights), ncols=len(widths),
                                sharex=True,
                                figsize=(widths[0], int(sum(heights))),
                                gridspec_kw=gs_kw)

        axs[0].plot(midpoints, label='midpoints', color='blue', alpha=0.6)
        axs[0].set_ylabel('Midpoint Price (USD)', color='black')

        # Redundant labeling for all computer compatibility
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

        axs[0].scatter(x=long_fills, y=midpoints[long_fills], label='buys', alpha=0.7,
                       color='green', marker="^")

        axs[0].scatter(x=short_fills, y=midpoints[short_fills], label='sells', alpha=0.7,
                       color='red', marker="v")

        axs[1].plot(inventory, label='inventory', color='orange')
        axs[1].axhline(0., color='grey')
        axs[1].set_ylabel('Inventory Count', color='black')
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
        # Redundant labeling for all computer compatibility
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
        plt.tight_layout()

        if save_filename is None:
            plt.show()
        else:
            plt.savefig(f"{save_filename}.png")
            plt.close(fig)

    def plot_obs(self, save_filename: str or None = None) -> None:
        """
        Represent all the observation spaces seen by the agent as one image.
        """
        observations = np.asarray(self._historical_observations, dtype=np.float32)
        plot_observation_space(observation=observations,
                               labels=self.observation_labels,
                               save_filename=save_filename)
