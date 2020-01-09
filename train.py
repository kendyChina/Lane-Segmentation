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

def compute_iou(pred, label, miou):
    """
    pred : [N, H, W]
    label: [N, H, W]
    """
    pred = pred.cpu().numpy()
    label = label.cpu().numpy()
    for i in range(CONFIG.NUM_CLASSES):
        single_label = label == i
        single_pred = pred == i
        temp_tp = np.sum(single_label * single_pred)
        temp_ta = np.sum(single_pred) + np.sum(single_label) - temp_tp
        miou["TP"][i] += temp_tp
        miou["TA"][i] += temp_ta
    return miou

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

def print_and_write(file, string):
    print(string)
    file.write(string + "\n")
    file.flush()

class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, file, patience=7, verbose=True, delta=0):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement.
                            Default: True
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
        """
        self.file = file
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta

    def __call__(self, val_loss, model, epoch):

        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model, epoch)
        elif score < self.best_score + self.delta:
            self.counter += 1
            print_and_write(self.file, f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model, epoch)
            self.counter = 0

    def save_checkpoint(self, val_loss, model, epoch):
        '''Saves model when validation loss decrease.'''
        if self.verbose:
            print_and_write(self.file,
                            f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), os.path.join(CONFIG.SAVE_PATH, 'checkpoint{}.pt'.format(epoch)))
        self.val_loss_min = val_loss

class Main(object):
    def __init__(self, network="UNet"):
        if "UNet" in network:
            self.model = UNet()
        elif "Resnet" in network:
            self.model = resnet18()
        else:
            raise ValueError("Network is not support")
        if CONFIG.CUDA_AVAIL:
            torch.cuda.set_device(CONFIG.SET_DEVICE)
            self.model = self.model.cuda()

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=CONFIG.BASE_LR, weight_decay=CONFIG.WEIGHT_DECAY)
        self.calc_loss = nn.CrossEntropyLoss()
        self.trainingF = open(os.path.join(CONFIG.SAVE_PATH, "training.csv"), mode="a")
        self.earlystop = EarlyStopping(self.trainingF, verbose=True)


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
        print_and_write(self.trainingF, "mask loss is {:.4f}".format(total_mask_loss / len(train_loader)))
        self.trainingF.flush()


    def valid_epoch(self):
        self.model.eval()
        total_mask_loss = 0.0
        dataprocess = tqdm(val_loader)
        dataprocess.set_description_str("epoch:{}".format(self.epoch))
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
        for i in range(1, CONFIG.NUM_CLASSES): # ignore class 0
            iou_i = iou["TP"][i] / iou["TA"][i]
            iou_string = "Class{}'s iou: {:.4f} \n".format(i, iou_i)
            total_iou += iou_i
            print(iou_string)
            self.trainingF.write(iou_string)
        print_and_write(self.trainingF, "MIoU is: {:.4f}".format(total_iou / CONFIG.NUM_CLASSES))

        avg_loss = total_mask_loss / len(val_loader)
        print_and_write(self.trainingF, "mask loss is {:.4f}".format(avg_loss))

        self.earlystop(avg_loss, self.model, self.epoch)
        self.trainingF.flush()


    def run(self):
        if os.path.exists(CONFIG.SAVE_PATH):
            self.model.apply(weights_init) # init
        else:
            self.model.load_state_dict(torch.load(os.path.join(CONFIG.SAVE_PATH, "checkpoint1.pt")))
        os.makedirs(CONFIG.SAVE_PATH, exist_ok=True)
        # self.valF = open(os.path.join(CONFIG.SAVE_PATH, "valid.csv"), mode="a")
        for epoch in range(1, CONFIG.EPOCHS + 1):
            print_and_write(self.trainingF, "********** EPOCH {} **********".format(epoch))
            self.epoch = epoch
            self.train_epoch()
            self.valid_epoch()
            if self.earlystop.early_stop:
                print_and_write("Early stopping")
                break
            # if epoch % CONFIG.SAVE_EPOCH:
            #     torch.save({'state_dict': self.model.state_dict()},
            #                os.path.join(os.getcwd(), CONFIG.SAVE_PATH, "laneModel{}.pth.tar".format(epoch)))
        self.trainingF.close()
        # self.testF.close()
        # torch.save({'state_dict': self.model.state_dict()},
        #            os.path.join(os.getcwd(), CONFIG.SAVE_PATH, "finalModel.pth.tar"))



if __name__ == '__main__':
    main = Main()
    main.run()