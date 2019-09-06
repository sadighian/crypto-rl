import matplotlib.pyplot as plt
import numpy as np


class TradingGraph:
    """
    A stock trading visualization using matplotlib
    made to render OpenAI gym environments
    """
    plt.style.use('dark_background')

    def __init__(self, sym=None):
        # attributes for rendering
        self.sym = sym
        self.line1 = []
        self.screen_size = 1000
        self.y_vec = None
        self.x_vec = np.linspace(0, self.screen_size * 10,
                                 self.screen_size + 1)[0:-1]

    def reset_render_data(self, y_vec):
        self.y_vec = y_vec
        self.line1 = []

    def render(self, midpoint=100., mode='human'):
        if mode == 'human':
            self.line1 = self.live_plotter(self.x_vec,
                                           self.y_vec,
                                           self.line1,
                                           identifier=self.sym)
            self.y_vec = np.append(self.y_vec[1:], midpoint)

    @staticmethod
    def live_plotter(x_vec, y1_data, line1, identifier='Add Symbol Name',
                     pause_time=0.00001):
        if not line1:
            # this is the call to matplotlib that allows dynamic plotting
            plt.ion()
            fig = plt.figure(figsize=(20, 12))
            ax = fig.add_subplot(111)
            # create a variable for the line so we can later update it
            line1, = ax.plot(x_vec, y1_data, '-', label='midpoint', alpha=0.8)
            # update plot label/title
            plt.ylabel('Price')
            plt.legend()
            plt.title('Title: {}'.format(identifier))
            plt.show(block=False)

        # after the figure, axis, and line are created, we only need to update the
        # y-data
        line1.set_ydata(y1_data)

        # adjust limits if new data goes beyond bounds
        if np.min(y1_data) <= line1.axes.get_ylim()[0] or \
                np.max(y1_data) >= line1.axes.get_ylim()[1]:
            plt.ylim(np.min(y1_data), np.max(y1_data))

        # this pauses the data so the figure/axis can catch up
        # - the amount of pause can be altered above
        plt.pause(pause_time)

        # return line so we can update it again in the next iteration
        return line1

    @staticmethod
    def close():
        plt.close()
