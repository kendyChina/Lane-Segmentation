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


class PlotImage(object):
    def __init__(self, win, env):
        self.vis = visdom.Visdom(port=CONFIG.PORT, env=env)
        self.win = win

    def __call__(self, img,  *args, **kwargs):
        img = np.array(img)
        if len(img.shape) == 3:
            img = img.transpose((2, 0, 1))  # (H, W, C) --> (C, H, W)
        elif len(img.shape) == 2:
            img = img[3, :]  # (H, W) --> (C, H, W)
        self.vis.image(img, win=self.win, opts=dict(title=self.win))


if __name__ == '__main__':
    plot_img = PlotImage("test1", "test-img")
    import torch
    x = torch.randn((3, 256, 96))
    plot_img(x)
