import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

from config import CONFIG
from models.UNet import UNet
from models.deeplabv3p_resnet import RESNETDeeplabV3Plus
from models.deeplabv3p_mobilenet import MobileNetDeeplabV3Plus
from utils.data_feeder import train_loader, val_loader
from utils.earlystop import PaW, EarlyStopping
from utils.metrics import compute_iou
from utils.visualize import PlotVisdom


def weights_init(m):
    if isinstance(m, (nn.Conv2d, nn.Linear)):
        nn.init.xavier_normal_(m.weight)
        nn.init.constant_(m.bias, 0.0)


def adjust_lr(optimizer, epoch):
    if epoch == 0:
        lr = 1e-3
    elif epoch == 2:
        lr = 1e-2
    elif epoch == 100:
        lr = 1e-3
    elif epoch == 150:
        lr = 1e-4
    else:
        return optimizer.param_groups[0]["lr"]
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    return lr


def get_lr_scheduler(optimizer):
    return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10, eta_min=2e-8)


class Main(object):

    def __init__(self, network="UNet"):
        network = network.lower()
        if "unet" in network:
            self.model = UNet()
        elif "deeplabv3p_resnet" in network:
            self.model = RESNETDeeplabV3Plus(pretrained=CONFIG.PRETRAIN)
        elif "deeplabv3p_mobilenet" in network:
            self.model = MobileNetDeeplabV3Plus(pretrained=CONFIG.PRETRAIN)
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

        self.train_losses = []
        self.valid_losses = []
        self.plot_loss = PlotVisdom("loss", ["train", "valid"])

        self.train_miou = []
        self.valid_miou = []
        self.plot_miou = PlotVisdom("miou", ["train", "valid"])

        self.plot_lr = PlotVisdom("LearningRate", "lr")

    def train_epoch(self):
        self.model.train()
        total_mask_loss = 0.0
        dataprocess = tqdm(train_loader)
        dataprocess.set_description_str("epoch:{}".format(self.epoch))
        iou = {"TP": {i: 0 for i in range(CONFIG.NUM_CLASSES)},
               "TA": {i: 0 for i in range(CONFIG.NUM_CLASSES)}}
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
            pred = torch.argmax(F.softmax(pred, dim=1), dim=1)
            iou = compute_iou(pred, label, iou)
            dataprocess.set_postfix_str("mask_loss:{:.4f}".format(total_mask_loss/ (batch_idx + 1)))

        total_iou = 0.0
        for i in range(CONFIG.NUM_CLASSES):
            iou_i = iou["TP"][i] / iou["TA"][i]
            iou_string = "Class{}'s iou: {:.4f}".format(i, iou_i)
            self.print_and_write(iou_string)
            if i > 0:  # ignore class 0
                total_iou += iou_i
        miou = total_iou / (CONFIG.NUM_CLASSES - 1)
        self.train_miou.append(miou)
        self.print_and_write("MIoU is: {:.4f} (without class 0)".format(miou))

        avg_loss = total_mask_loss / len(train_loader)
        self.train_losses.append(avg_loss)
        self.print_and_write("mask loss is {:.4f}".format(avg_loss))

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

        miou = total_iou / (CONFIG.NUM_CLASSES - 1)
        self.valid_miou.append(miou)
        self.print_and_write("MIoU is: {:.4f} (without class 0)".format(miou))

        avg_loss = total_mask_loss / len(val_loader)
        self.valid_losses.append(avg_loss)
        self.print_and_write("mask loss is {:.4f}".format(avg_loss))

        # self.earlystop(avg_loss, self.model)
        self.trainingF.flush()

    def run(self):
        checkpoint_path = os.path.join(CONFIG.SAVE_PATH, CONFIG.CHECKPOINT_FILE)
        if os.path.exists(checkpoint_path):
            self.model.load_state_dict(torch.load(checkpoint_path))
            self.print_and_write("load checkpoint succeed")
        # else:
            # self.model.apply(weights_init)  # init
            # self.print_and_write("init weights succeed")
        self.print_and_write("Parameters:\n"
                             "\tBatch size: {}\n"
                             "\tImage size: {}\n".format(CONFIG.BATCH_SIZE, CONFIG.IMG_SIZE, CONFIG.BASE_LR))
        lr_list = []
        scheduler = get_lr_scheduler(self.optimizer)
        for epoch in range(0, CONFIG.EPOCHS + 0):
            self.print_and_write("********** EPOCH {} **********".format(epoch))
            lr = scheduler.get_lr()[0]
            self.print_and_write("Learning rate: {}".format(lr))
            lr_list.append(lr)
            self.epoch = epoch
            self.train_epoch()
            self.valid_epoch()
            scheduler.step(epoch)
            self.plot_loss(self.train_losses, self.valid_losses)
            self.plot_miou(self.train_miou, self.valid_miou)
            self.plot_lr(lr_list)
            torch.save(self.model.state_dict(), os.path.join(CONFIG.SAVE_PATH, CONFIG.CHECKPOINT_FILE))
            # if self.earlystop.early_stop:
            #     self.print_and_write("Early stopping")
            #     break
            # self.print_and_write("\n")
        self.trainingF.close()


if __name__ == '__main__':
    main = Main(network=CONFIG.MODEL)
    main.run()