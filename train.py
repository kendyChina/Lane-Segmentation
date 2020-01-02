import numpy as np
import torch
import torch.nn as nn
import pandas as pd
from utils.data_feeder import dataprocess, cuda_avail
from models.UNet import UNet
from models.ResNet import resnet18

if cuda_avail:
    torch.cuda.set_device(6)

# MIoU: Mean Intersection over Union
class MeanIoU(object):
    def __init__(self, num_classes=8):
        self.num_classes = num_classes
        # self.confusionMatrix = np.zeros((self.num_classes, ) * 2)
    def getConfusionMatrix(self):
        # print(pred)
        # print(label)
        # Return True and False
        mask = (self.label >= 0) & (self.label < self.num_classes)
        # print(mask)
        # print(label[mask])
        # print(pred[mask])
        label = self.num_classes * self.label[mask] + self.pred[mask]
        # print(label)
        count = np.bincount(label, minlength=self.num_classes ** 2)
        # print(count)
        confusionMatrix = count.reshape(self.num_classes, self.num_classes)
        # print(confusionMatrix)
        self.confusionMatrix = confusionMatrix

    def meanIntersectionOverUnion(self):
        # IoU = TP / (TP + FP + FN) = intersection / union
        intersection = np.diag(self.confusionMatrix) # 取对角线值，返回list
        # print(intersection)
        union = np.sum(self.confusionMatrix, axis=1) + np.sum(self.confusionMatrix, axis=0)
        # print(union)
        IoU = []
        for i, val in enumerate(intersection):
            if union[i] != 0:
                IoU.append(intersection[i] / union[i])
            else:
                IoU.append(1.0)
        # print(IoU)
        mIoU = np.nanmean(IoU) # Compute the mean of each IoU
        # print(mIoU)
        return mIoU

    def __call__(self, pred, label):
        print(pred.shape)
        print(label.shape)
        # assert pred.shape[-2:] == label.shape[-2:]
        self.pred = pred
        self.label = label
        self.getConfusionMatrix()
        return self.meanIntersectionOverUnion()

class Loss(object):
    def __call__(self, pred ,label):
        pred = torch.from_numpy(pred.astype(np.float32))
        label = torch.from_numpy(label.astype(np.float32))
        ce_loss = nn.CrossEntropyLoss()
        res_ce_loss = ce_loss(pred, label)
        print(res_ce_loss)

def main():
    num_classes = 8
    miou = MeanIoU(num_classes)
    loss = Loss()
    model = UNet()
    if cuda_avail:
        model.cuda()
    model.eval()
    for batch_item in dataprocess:
        image, label = batch_item["image"], batch_item["label"]
        if cuda_avail:
            image, label = image.cuda(), label.cuda()
        pred = model(image)
        res_miou = miou(pred, label)
        print(res_miou)



if __name__ == '__main__':
    # miou = MeanIoU()
    # pred = np.array([[-1, 0, 1], [3, 4, 10]])
    # label = np.array([[-1, 1, 1], [3, 4, 10]])
    # miou(pred, label)
    # loss = Loss()
    # loss(pred, label)
    main()