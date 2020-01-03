import numpy as np
import time
import torch
from utils.data_feeder import dataprocess, cuda_avail
from utils.loss.focal_loss import FocalLoss
from utils.loss.dice_loss import DC_and_CE_loss
from models.UNet import UNet
from models.ResNet import resnet18

if cuda_avail:
    torch.cuda.set_device(2)

# MIoU: Mean Intersection over Union
class MeanIoU(object):
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
        self.num_classes = pred.shape[1]
        pred = pred.cpu()
        label = label.cpu()
        pred = torch.argmax(pred, dim=1)
        # print(pred.shape)
        # print(label.shape)
        assert pred.shape[-2:] == label.shape[-2:]
        self.pred = pred
        self.label = label
        self.getConfusionMatrix()
        return self.meanIntersectionOverUnion()

class Loss(object):
    def __call__(self, pred ,label):
        # pred = torch.from_numpy(pred.astype(np.float32))
        # label = torch.from_numpy(label.astype(np.float32))
        # ce_loss = nn.CrossEntropyLoss()
        # focal_loss = FocalLoss()
        # res_ce_loss = focal_loss(pred, label)
        dc_ce_loss = DC_and_CE_loss({}, {})
        res_dc_ce_loss = dc_ce_loss(pred, label)
        return res_dc_ce_loss

class Network(object):
    def __init__(self, is_train=True):
        self.is_train = is_train
    def __call__(self, image, label, network="UNet"):
        if "UNet" in network:
            model = UNet()
        elif "Resnet" in network:
            model = resnet18()

        if self.is_train:
            model.train()
        else:
            model.eval()
        if cuda_avail:
            model.cuda()
        pred = model(image)

        create_miou = MeanIoU()
        miou = create_miou(pred, label)
        create_loss = Loss()
        loss = create_loss(pred, label)
        return loss, miou, pred, model

def main():
    epochs = 2
    iter_id = 0
    for epoch in range(epochs):
        for batch_item in dataprocess:
            time_1 = time.time()
            iter_id += 1
            image, label = batch_item["image"], batch_item["label"]
            if cuda_avail:
                image, label = image.cuda(), label.cuda()
            run_network = Network(is_train=True)
            loss, miou, pred, model = run_network(image, label, network="UNet")
            optimizer = torch.optim.Adam(model.parameters())
            optimizer.step()
            time_2 = time.time()
            print("Iter - {}: train loss: {:.6f}, mean iou: {:.6f}, "
                  "cost time: {:.6f}".format(iter_id, loss, miou, time_2 - time_1))


if __name__ == '__main__':
    # miou = MeanIoU()
    # pred = np.array([[-1, 0, 1], [3, 4, 10]])
    # label = np.array([[-1, 1, 1], [3, 4, 10]])
    # miou(pred, label)
    # loss = Loss()
    # loss(pred, label)
    main()