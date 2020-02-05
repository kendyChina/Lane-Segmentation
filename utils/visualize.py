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
        self.vis.image(np.array(img).transpose((2, 0, 1)),  # (H, W, C) --> (C, H, W)
                       win=self.win, opts=dict(caption=self.win, title=self.win))


class PlotHeatmap(object):
    def __init__(self, win, env, **opts):
        self.vis = visdom.Visdom(port=CONFIG.PORT, env=env)
        self.win = win
        self.opts = opts
    def __call__(self, hm, *args, **kwargs):
        hm = np.flipud(hm)
        self.vis.heatmap(hm, opts=self.opts)


if __name__ == '__main__':
    plot = PlotHeatmap("test", "test-env", rownames=[i for i in range(8)], columnnames=[i for i in range(8)],
                       xmin=0,colormap="b")
    import torch
    from utils.metrics import ComputeIoU
    a = torch.tensor([1,5,7,3,4])
    b = torch.tensor([1,4,7,3,5])
    compute = ComputeIoU()
    compute(a, b)
    cfs = compute.get_cfsmatrix()
    plot(cfs)

