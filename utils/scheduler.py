import torch
import torch.nn as nn
from torch.optim.lr_scheduler import ReduceLROnPlateau

class A(nn.Module):
    def __init__(self):
        super(A, self).__init__()
        self.conv1 = nn.Conv2d(3, 3, 3)
        self.conv2 = nn.Conv2d(3, 3, 3)
    def forward(self, x):
        return self.conv2(self.conv1(x))

class MyReduceLROnPlateau(ReduceLROnPlateau):

    def __init__(self, optimizer, mode='min', factor=0.1, patience=10,
                 verbose=False, threshold=1e-4, threshold_mode='rel',
                 cooldown=0, min_lr=0, eps=1e-8):
        super(MyReduceLROnPlateau, self).__init__(optimizer=optimizer, mode=mode, factor=factor, patience=patience,
                 verbose=verbose, threshold=threshold, threshold_mode=threshold_mode,
                 cooldown=cooldown, min_lr=min_lr, eps=eps)
        self.new_lr = optimizer.param_groups[0]["lr"]

    def _reduce_lr(self, epoch):
        for i, param_group in enumerate(self.optimizer.param_groups):
            old_lr = float(param_group['lr'])
            new_lr = max(old_lr * self.factor, self.min_lrs[i])
            self.new_lr = new_lr
            if old_lr - new_lr > self.eps:
                param_group['lr'] = new_lr
                if self.verbose:
                    print('Epoch {:5d}: reducing learning rate'
                          ' of group {} to {:.4e}.'.format(epoch, i, new_lr))

    def get_lr(self):
        return [self.new_lr,]


if __name__ == '__main__':
    model = A()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    scheduler = MyReduceLROnPlateau(optimizer, patience=1, verbose=True)