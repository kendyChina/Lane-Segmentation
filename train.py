import os
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

from config import CONFIG
from models.UNet import UNet
from utils.data_feeder import train_loader, val_loader
from utils.earlystop import PaW, EarlyStopping

checkpoint_file = "checkpoint_unet.pt"


def now():
    return time.strftime("%Y-%m-%d %H:%M:%S ", time.localtime())


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
        self.num_classes = CONFIG.NUM_CLASSES
        pred = pred.cpu()
        label = label.cpu()
        pred = torch.argmax(F.softmax(pred, dim=1), dim=1)
        # print(pred.shape)
        # print(label.shape)
        assert pred.shape[-2:] == label.shape[-2:]
        self.pred = pred
        self.label = label
        self.getConfusionMatrix()
        return self.meanIntersectionOverUnion()

def compute_iou(pred, label, iou):
    """
    pred : [N, H, W]
    label: [N, H, W]
    """
    pred = pred.cpu().numpy()
    label = label.cpu().numpy()
    for i in range(CONFIG.NUM_CLASSES):
        single_pred = pred == i
        single_label = label == i
        temp_tp = np.sum(single_pred * single_label)
        temp_ta = np.sum(single_pred) + np.sum(single_label) - temp_tp
        iou["TP"][i] += temp_tp
        iou["TA"][i] += temp_ta
    return iou


def weights_init(m):
    if isinstance(m, (nn.Conv2d, nn.Linear)):
        nn.init.xavier_normal_(m.weight)
        nn.init.constant_(m.bias, 0.0)

class Main(object):
    def __init__(self, network="UNet"):
        if "UNet" in network:
            self.model = UNet()
        else:
            raise ValueError("Network is not support")
        if CONFIG.CUDA_AVAIL:
            torch.cuda.set_device(CONFIG.SET_DEVICE)
            self.model = self.model.cuda()

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=CONFIG.BASE_LR, weight_decay=CONFIG.WEIGHT_DECAY)
        self.calc_loss = nn.CrossEntropyLoss()
        self.trainingF = open(os.path.join(CONFIG.SAVE_PATH, "training.csv"), mode="a")
        self.earlystop = EarlyStopping(self.trainingF, verbose=True)
        self.print_and_write = PaW(self.trainingF)


    def train_epoch(self):
        self.model.train()
        total_mask_loss = 0.0
        dataprocess = tqdm(train_loader)
        dataprocess.set_description_str("epoch:{}".format(self.epoch))
        for batch_idx, batch_item in enumerate(dataprocess):
            image, label = batch_item["image"], batch_item["label"]
            if CONFIG.CUDA_AVAIL:
                image, label = image.cuda(), label.cuda()
            self.optimizer.zero_grad()
            pred = self.model(image)
            mask_loss = self.calc_loss(pred, label)
            total_mask_loss += mask_loss.item()
            mask_loss.backward()
            self.optimizer.step()
            dataprocess.set_postfix_str("mask_loss:{:.4f}".format(total_mask_loss/ (batch_idx + 1)))
        self.print_and_write("mask loss is {:.4f}".format(total_mask_loss / len(train_loader)))
        self.trainingF.flush()


    def valid_epoch(self):
        self.model.eval()
        total_mask_loss = 0.0
        dataprocess = tqdm(val_loader)
        dataprocess.set_description_str("epoch:{}".format(self.epoch))
        # iou is for sum TP and TA of each class
        iou = {"TP": {i: 0 for i in range(CONFIG.NUM_CLASSES)},
               "TA": {i: 0 for i in range(CONFIG.NUM_CLASSES)}}
        for batch_idx, batch_item in enumerate(dataprocess):
            image, label = batch_item["image"], batch_item["label"]
            if CONFIG.CUDA_AVAIL:
                image, label = image.cuda(), label.cuda()
            pred = self.model(image)
            mask_loss = self.calc_loss(pred, label)
            total_mask_loss += mask_loss.detach().item()
            pred = torch.argmax(F.softmax(pred, dim=1), dim=1)
            iou = compute_iou(pred, label, iou)
            dataprocess.set_postfix_str("mask_loss:{:.4f}".format(total_mask_loss / (batch_idx + 1)))

        total_iou = 0.0
        for i in range(CONFIG.NUM_CLASSES):
            iou_i = iou["TP"][i] / iou["TA"][i]
            iou_string = "Class{}'s iou: {:.4f}".format(i, iou_i)
            self.print_and_write(iou_string)
            if i > 0: # ignore class 0
                total_iou += iou_i
        self.print_and_write("MIoU is: {:.4f} (ignore class 0)".format(total_iou / (CONFIG.NUM_CLASSES - 1)))

        avg_loss = total_mask_loss / len(val_loader)
        self.print_and_write("mask loss is {:.4f}".format(avg_loss))

        self.earlystop(avg_loss, self.model)
        self.trainingF.flush()


    def run(self):
        checkpoint_path = os.path.join(CONFIG.SAVE_PATH, checkpoint_file)
        if os.path.exists(checkpoint_path):
            self.model.load_state_dict(torch.load(checkpoint_path))
            self.print_and_write("load checkpoint succeed")
        else:
            self.model.apply(weights_init)  # init
            self.print_and_write("init weights succeed")
        for epoch in range(28, CONFIG.EPOCHS + 28):
            self.print_and_write("********** EPOCH {} **********".format(epoch))
            self.epoch = epoch
            self.train_epoch()
            self.valid_epoch()
            if self.earlystop.early_stop:
                self.print_and_write("Early stopping")
                break
            self.print_and_write("\n")
        self.trainingF.close()


if __name__ == '__main__':
    main = Main()
    main.run()