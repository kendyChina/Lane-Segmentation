import visdom
import numpy as np


port = 8888


class PlotVisdom(object):
    def __init__(self, win, legends):
        self.vis = visdom.Visdom(port=port)
        self.win = win
        self.legends = legends

    def __call__(self, *args):
        args_ = [np.array(i) for i in args]
        x_points = np.arange(len(args[0]))
        Y = np.column_stack(tuple(args_))
        X = np.column_stack(tuple([x_points for _ in range(len(args))]))
        Y = np.squeeze(Y) # squeeze the dimension when len(args)==1
        X = np.squeeze(X)
        self.vis.line(
            Y=Y, X=X,
            win=self.win,
            opts=dict(showlegend=True, title=self.win, legend=self.legends)
        )


if __name__ == '__main__':
    plot_loss = PlotVisdom("loss")
