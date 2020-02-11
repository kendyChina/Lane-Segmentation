import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

from config import CONFIG
from models.UNet import UNet
from models.UNet_MixNet import MixNetUNet
from models.deeplabv3p_resnet import RESNETDeeplabV3Plus
from models.deeplabv3p_mobilenet import MobileNetDeeplabV3Plus
from utils.data_feeder import train_loader, val_loader
from utils.earlystop import PaW, EarlyStopping
from utils.metrics import ComputeIoU
from utils.visualize import PlotVisdom, PlotCfsMatrix
from utils.scheduler import MyReduceLROnPlateau


def weights_init(m):
    if isinstance(m, (nn.Conv2d, nn.Linear)):
        nn.init.xavier_normal_(m.weight)
        nn.init.constant_(m.bias, 0.0)


def adjust_lr(optimizer, epoch):
    # if epoch == 0:
    #     lr = 1e-3
    # elif epoch == 2:
    #     lr = 1e-2
    # elif epoch == 100:
    #     lr = 1e-3
    # elif epoch == 150:
    #     lr = 1e-4
    # else:
    #     return optimizer.param_groups[0]["lr"]
    lr = optimizer.param_groups[0]["lr"]
    cos_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,
                                                               T_max=CONFIG.CosineT_max, eta_min=2e-8)
    if epoch <= 30:
        lr = cos_scheduler.get_lr()[0]
        cos_scheduler.step(epoch)

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    return lr


def get_optimizer(model, lr, weight_decay):
    return torch.optim.Adam(params=model.parameters(),
                            lr=lr, weight_decay=weight_decay)


def get_loss_fn():
    return nn.CrossEntropyLoss()


def get_lr_scheduler(optimizer):
    # return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=CONFIG.CosineT_max, eta_min=2e-8)
    return MyReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=2, verbose=False,
                                                      threshold_mode="rel", threshold=1e-3, cooldown=0,
                                                      min_lr=0, eps=1e-8)


class Main(object):

    def __init__(self, network="UNet"):
        network = network.lower()
        if "unet_base" in network:
            self.model = UNet(batch_norm=True, bias=False)
        elif "unet_mixnet" in network:
            self.model = MixNetUNet(batch_norm=True, bias=False)
        elif "deeplabv3p_resnet" in network:
            self.model = RESNETDeeplabV3Plus(pretrained=CONFIG.PRETRAIN)
        elif "deeplabv3p_mobilenet" in network:
            self.model = MobileNetDeeplabV3Plus(pretrained=CONFIG.PRETRAIN)
        else:
            raise ValueError("Network is not support")
        if CONFIG.CUDA_AVAIL:
            torch.cuda.set_device(CONFIG.CUDA_DEVICE)
            self.model = self.model.cuda()

        self.optimizer = get_optimizer(self.model, lr=CONFIG.BASE_LR, weight_decay=CONFIG.WEIGHT_DECAY)
        self.calc_loss = get_loss_fn()
        self.trainingF = open(os.path.join(CONFIG.SAVE_PATH, CONFIG.LOGGING_FILE), mode="a")
        self.earlystop = EarlyStopping(self.trainingF, verbose=True)
        self.print_and_write = PaW(self.trainingF)

        self.train_losses = []
        self.valid_losses = []
        self.plot_loss = PlotVisdom("loss", ["train", "valid"])

        self.train_miou = []
        self.valid_miou = []
        self.plot_miou = PlotVisdom("miou", ["train", "valid"])

        self.val_ious = [[] for _ in range(CONFIG.NUM_CLASSES)]
        self.plot_ious = PlotVisdom("iou", ["class{}".format(i) for i in range(CONFIG.NUM_CLASSES)])

        self.plot_lr = PlotVisdom("LearningRate", ["lr",])
        self.plot_cfsmatrix = PlotCfsMatrix("ConfusionMatrix", CONFIG.ENV)

    def train_epoch(self):
        self.model.train()
        compute_iou = ComputeIoU()
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
            pred = torch.argmax(F.softmax(pred, dim=1), dim=1)
            compute_iou(pred, label)
            dataprocess.set_postfix_str("mask_loss:{:.4f}".format(total_mask_loss/ (batch_idx + 1)))

        ious = compute_iou.get_ious()
        for cls, iou in ious.items():
            iou_string = "Class{}'s iou: {:.4f}".format(cls, iou)
            self.print_and_write(iou_string)
        miou = compute_iou.get_miou(ignore=CONFIG.IGNORE)
        self.train_miou.append(miou)
        self.print_and_write("MIoU is: {:.4f} (without class 0)".format(miou))

        avg_loss = total_mask_loss / len(train_loader)
        self.train_losses.append(avg_loss)
        self.print_and_write("mask loss is {:.4f}".format(avg_loss))

        self.trainingF.flush()

    def valid_epoch(self):
        with torch.no_grad():
            self.model.eval()
            compute_iou = ComputeIoU()
            total_mask_loss = 0.0
            dataprocess = tqdm(val_loader)
            dataprocess.set_description_str("epoch:{}".format(self.epoch))


            for batch_idx, batch_item in enumerate(dataprocess):
                image, label = batch_item["image"], batch_item["label"]
                if CONFIG.CUDA_AVAIL:
                    image, label = image.cuda(), label.cuda()
                pred = self.model(image)
                mask_loss = self.calc_loss(pred, label)
                total_mask_loss += mask_loss.detach().item()

                pred = torch.argmax(F.softmax(pred, dim=1), dim=1)
                compute_iou(pred, label)
                dataprocess.set_postfix_str("mask_loss:{:.4f}".format(total_mask_loss / (batch_idx + 1)))

            ious = compute_iou.get_ious()
            for cls, iou in ious.items():
                iou_string = "Class{}'s iou: {:.4f}".format(cls, iou)
                self.print_and_write(iou_string)
                self.val_ious[cls].append(iou)
            miou = compute_iou.get_miou(ignore=CONFIG.IGNORE)
            self.plot_cfsmatrix(compute_iou.get_cfsmatrix())
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
            checkpoint = torch.load(checkpoint_path)
            if "state_dict" in checkpoint:
                self.model.load_state_dict(checkpoint["state_dict"])
                resume_epoch = checkpoint["epoch"] + 1
                self.optimizer = checkpoint["optimizer"]
                scheduler = checkpoint["scheduler"]
                self.train_losses = checkpoint["train_losses"]
                self.valid_losses = checkpoint["valid_losses"]
                self.train_miou = checkpoint["train_miou"]
                self.valid_miou = checkpoint["valid_miou"]
                self.val_ious = checkpoint["val_ious"]
                lr_list = checkpoint["lr_list"]
            else:
                self.model.load_state_dict(checkpoint)
                resume_epoch = 0
                lr_list = []
                scheduler = get_lr_scheduler(self.optimizer)
            self.print_and_write("load checkpoint succeed")
        else:
            resume_epoch = 0
            lr_list = []
            scheduler = get_lr_scheduler(self.optimizer)
        self.print_and_write("Parameters:\n"
                             "\tBatch size: {}\n"
                             "\tImage size: {}\n".format(CONFIG.BATCH_SIZE, CONFIG.IMG_SIZE, CONFIG.BASE_LR))
        for epoch in range(resume_epoch, CONFIG.EPOCHS + resume_epoch):
            self.print_and_write("********** EPOCH {} **********".format(epoch))
            lr = scheduler.get_lr()[0]
            # lr = adjust_lr(self.optimizer, epoch)
            self.print_and_write("Learning rate: {}".format(lr))
            lr_list.append(lr)
            self.epoch = epoch
            self.train_epoch()
            self.valid_epoch()
            # scheduler.step(epoch)
            scheduler.step(self.valid_losses[-1])

            self.plot_loss(self.train_losses, self.valid_losses)
            self.plot_miou(self.train_miou, self.valid_miou)
            self.plot_ious(*self.val_ious)
            self.plot_lr(lr_list)

            state = {"epoch": epoch,
                     "state_dict": self.model.state_dict(),
                     "optimizer": self.optimizer.state_dict(),
                     "scheduler": scheduler.state_dict(),
                     "train_losses": self.train_losses,
                     "valid_losses": self.valid_losses,
                     "train_miou": self.train_miou,
                     "valid_miou": self.valid_miou,
                     "val_ious": self.val_ious,
                     "lr_list": lr_list}
            torch.save(state, os.path.join(CONFIG.SAVE_PATH, CONFIG.CHECKPOINT_FILE))

            # if self.earlystop.early_stop:
            #     self.print_and_write("Early stopping")
            #     break
            # self.print_and_write("\n")
        self.trainingF.close()


if __name__ == '__main__':
    import numpy as np
    import torch
    main = Main(network=CONFIG.MODEL)
    main.run()