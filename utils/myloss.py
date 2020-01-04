import torch
import torch.nn as nn

class BinaryDiceLoss(nn.Module):
    def __init__(self, smooth=1, p=2, reduction="mean"):
        super(BinaryDiceLoss, self).__init__()
        self.smooth = smooth
        self.p = p
        self.reduction = reduction

    def forward(self, pred, target):
        assert pred.shape[0] == target.shape[0] # Same batch-size
        pred = pred.contiguous().view(pred.shape[0], -1)
        target = target.contiguous().view(target.shape[0], -1)

        num = 2 * torch.sum(torch.mul(pred, target), dim=1) + self.smooth
        den = torch.sum(pred.pow(self.p) + target.pow(self.p), dim=1) + self.smooth

        loss = 1 - num / den

        if self.reduction == "mean":
            return loss.mean()
        if self.reduction == "sum":
            return loss.sum()
        if self.reduction == "none":
            return loss
        raise ValueError("reduction is not support")

