import visdom
import numpy as np
from config import CONFIG


class PlotVisdom(object):
    def __init__(self, win, legends):
        self.vis = visdom.Visdom(port=CONFIG.PORT, env=CONFIG.ENV)
        self.win = win
        self.legends = legends
        self.lines = len(legends)
        self.env = CONFIG.ENV
        self.port = CONFIG.PORT

    def __call__(self, *args):
        if self.lines == 1:
            Y = np.array(args[0])
            X = np.arange(len(Y))
        else:
            args_ = [np.array(i) for i in args]
            x_points = np.arange(len(args[0]))
            Y = np.column_stack(tuple(args_))
            X = np.column_stack(tuple([x_points for _ in range(len(args))]))

        self.vis.line(
            Y=Y, X=X,
            win=self.win,
            opts=dict(showlegend=True, title=self.win, legend=self.legends)
        )


if __name__ == '__main__':
    plot_loss = PlotVisdom("loss", [1])
