import os
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

from config import CONFIG
from models.UNet import UNet
from models.deeplabv3p import DeeplabV3Plus
from utils.data_feeder import train_loader, val_loader
from utils.earlystop import PaW, EarlyStopping
from utils.metrics import compute_iou


def weights_init(m):
    if isinstance(m, (nn.Conv2d, nn.Linear)):
        nn.init.xavier_normal_(m.weight)
        nn.init.constant_(m.bias, 0.0)


class Main(object):
    def __init__(self, network="UNet"):
        network = network.lower()
        if "unet" in network:
            self.model = UNet()
        elif "deeplab" in network:
            self.model = DeeplabV3Plus(pretrained=CONFIG.PRETRAIN)
        else:
            raise ValueError("Network is not support")
        if CONFIG.CUDA_AVAIL:
            torch.cuda.set_device(CONFIG.CUDA_DEVICE)
            self.model = self.model.cuda()

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=CONFIG.BASE_LR, weight_decay=CONFIG.WEIGHT_DECAY)
        self.calc_loss = nn.CrossEntropyLoss()
        self.trainingF = open(os.path.join(CONFIG.SAVE_PATH, CONFIG.LOGGING_FILE), mode="a")
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
        self.print_and_write("MIoU is: {:.4f} (without class 0)".format(total_iou / (CONFIG.NUM_CLASSES - 1)))

        avg_loss = total_mask_loss / len(val_loader)
        self.print_and_write("mask loss is {:.4f}".format(avg_loss))

        self.earlystop(avg_loss, self.model)
        self.trainingF.flush()


    def run(self):
        checkpoint_path = os.path.join(CONFIG.SAVE_PATH, CONFIG.CHECKPOINT_FILE)
        if os.path.exists(checkpoint_path):
            self.model.load_state_dict(torch.load(checkpoint_path))
            self.print_and_write("load checkpoint succeed")
        else:
            self.model.apply(weights_init)  # init
            self.print_and_write("init weights succeed")
        self.print_and_write("Parameters:\n"
                             "\tBatch size: {}\n"
                             "Image size: {}\n"
                             "Learning rate: {}\n")
        for epoch in range(1, CONFIG.EPOCHS + 1):
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
    main = Main(network=CONFIG.MODEL)
    main.run()