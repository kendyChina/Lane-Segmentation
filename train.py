import numpy as np
import os, shutil, time
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.data_feeder import train_loader, val_loader
from utils.loss.focal_loss import FocalLoss
from utils.loss.focal_loss import FocalLoss
from utils.loss.dice_loss import DC_and_CE_loss
from models.UNet import UNet
from models.ResNet import resnet18
from config import CONFIG

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

def compute_miou(pred, label, result):
    """
        pred : [N, H, W]
        label: [N, H, W]
        """
    pred = pred.cpu().numpy()
    label = label.cpu().numpy()
    for i in range(CONFIG.NUM_CLASSES7):
        single_label = label == i
        single_pred = pred == i
        temp_tp = np.sum(single_label * single_pred)
        temp_ta = np.sum(single_pred) + np.sum(single_label) - temp_tp
        result["TP"][i] += temp_tp
        result["TA"][i] += temp_ta
    return result

class Loss(object):
    def __call__(self, pred ,label):
        # pred = torch.from_numpy(pred.astype(np.float32))
        # label = torch.from_numpy(label.astype(np.float32))
        # ce_loss = nn.CrossEntropyLoss()
        # focal_loss = FocalLoss()
        # res_ce_loss = focal_loss(pred, label)
        # dc_ce_loss = DC_and_CE_loss({}, {})
        # res_dc_ce_loss = dc_ce_loss(pred, label)
        res = nn.CrossEntropyLoss(reduction="mean")(pred, label)
        return res

def weights_init(m):
    if isinstance(m, (nn.Conv2d, nn.Linear)):
        nn.init.xavier_normal_(m.weight)
        nn.init.constant_(m.bias, 0.0)

class Main(object):
    def __init__(self, network="UNet", reflash_log=False):
        if "UNet" in network:
            self.model = UNet()
        elif "Resnet" in network:
            self.model = resnet18()
        else:
            raise ValueError("Network is not support")
        if CONFIG.CUDA_AVAIL:
            torch.cuda.set_device(CONFIG.SET_DEVICE)
            self.model.cuda()
        self.model.apply(weights_init)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=CONFIG.BASE_LR, weight_decay=CONFIG.WEIGHT_DECAY)
        self.calc_loss = nn.CrossEntropyLoss()

        self.reflash_log = reflash_log


    def train_epoch(self):
        self.model.train()
        total_mask_loss = 0.0
        dataprocess = tqdm(train_loader)
        dataprocess.set_description_str("epoch:{}".format(self.epoch))
        for batch_item in dataprocess:
            image, label = batch_item["image"], batch_item["label"]
            if CONFIG.CUDA_AVAIL:
                image.cuda(), label.cuda()
            self.optimizer.zero_grad()
            pred = self.model(image)
            mask_loss = self.calc_loss(pred, label)
            total_mask_loss += mask_loss.item()
            mask_loss.backward()
            self.optimizer.step()
            dataprocess.set_postfix_str("mask_loss:{:.4f}".format(mask_loss.item()))
        self.trainF.write("Epoch:{}, mask loss is {:.4f} \n".format(self.epoch, total_mask_loss / len(train_loader)))
        self.trainF.flush()


    def test(self):
        self.model.eval()
        total_mask_loss = 0.0
        dataprocess = tqdm(val_loader)
        dataprocess.set_description_str("epoch:{}".format(self.epoch))
        result = {"TP": {i: 0 for i in range(CONFIG.NUM_CLASSES)},
                  "TA": {i: 0 for i in range(CONFIG.NUM_CLASSES)}}
        for batch_item in dataprocess:
            image, label = batch_item["image"], batch_item["label"]
            if CONFIG.CUDA_AVAIL:
                image.cuda(), label.cuda()
            pred = self.model(image)
            mask_loss = self.calc_loss(pred, label)
            total_mask_loss += mask_loss.detach().item()
            pred


    def run(self):
        if os.path.exists(CONFIG.SAVE_PATH):
            if self.reflash_log:
                shutil.rmtree(CONFIG.SAVE_PATH)
        os.makedirs(CONFIG.SAVE_PATH, exist_ok=True)
        self.trainF = open(os.path.join(CONFIG.SAVE_PATH, "train.csv"), mode="w")
        self.testF = open(os.path.join(CONFIG.SAVE_PATH, "test.csv"), mode="w")
        for epoch in CONFIG.EPOCHS:
            self.epoch = epoch
            self.train_epoch()
            self.test()
        self.trainF.close()
        self.testF.close()



def main():
    network = "UNet"

    if "UNet" in network:
        model = UNet()
    elif "Resnet" in network:
        model = resnet18()
    else:
        raise ValueError("Network is not support")

    if CONFIG.CUDA_AVAIL:
        torch.cuda.set_device(CONFIG.SET_DEVICE)
        model.cuda()

    model.apply(weights_init)

    optimizer = torch.optim.Adam(model.parameters())
    calc_miou = MeanIoU()
    calc_loss = Loss()

    train_len = len(train_loader)
    for epoch in range(CONFIG.EPOCHS):
        prev_loss = 0.0
        total_loss = 0.0
        iter_id = 0
        model.train()
        train_loader.set_description_str("epoch: {}".format(epoch))
        time_1 = time.time()
        for batch_item in train_loader:
            optimizer.zero_grad()
            iter_id += 1
            image, label = batch_item["image"], batch_item["label"]
            if CONFIG.CUDA_AVAIL:
                image, label = image.cuda(), label.cuda()
            pred = model(image)
            miou = calc_miou(pred, label)
            this_loss = calc_loss(pred, label)
            this_loss.backward()
            optimizer.step()
            total_loss += this_loss.item()
            train_loader.set_postfix_str("train_loss: {:.4f}".format(total_loss / iter_id))
            # print("avg_loss: {:.10f}, miou: {:.10f}".format(total_loss / iter_id, miou))
        time_2 = time.time()
        train_time = time_2 - time_1
        train_epoch_str = "Epoch: {}, train_loss is {:.4f}, miou is {:.4f}".format(epoch, total_loss / train_len, )
        print(train_epoch_str)

            # model.eval()
            # pred = model(image)

            # print_msg = (f'[{iter_id}/{total_iter}] ' +
            #              f'train_loss: {train_loss:.5f} ' +
            #              f'valid_loss: {valid_loss:.5f}')
            # print(print_msg)
            # print("Iter[{}/{}]: train loss: {:.20f}, mean iou: {:.6f}, "
            #       "cost time: {:.6f}".format(iter_id, total_iter, this_loss, miou, time_2 - time_1))


if __name__ == '__main__':
    # miou = MeanIoU()
    # pred = np.array([[-1, 0, 1], [3, 4, 10]])
    # label = np.array([[-1, 1, 1], [3, 4, 10]])
    # miou(pred, label)
    # loss = Loss()
    # loss(pred, label)
    main()