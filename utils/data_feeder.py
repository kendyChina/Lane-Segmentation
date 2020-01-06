import torch
from torchvision import transforms
from torch.utils.data import DataLoader
from utils.image_process import LaneDataset, ImageAug, DeformAug, ToTensor
from config import CONFIG

print("cuda is available: {}".format(CONFIG.CUDA_AVAIL))
# pin_memory: 锁页内存
kwargs = {"num_workers": 1, "pin_memory": False} if CONFIG.CUDA_AVAIL else {}
train_dir = r"data_list/train.csv"
val_dir = r"data_list/val.csv"
test_dir = r"data_list/test.csv"
# train_dir = r"../data_list/train.csv"
# val_dir = r"../data_list/val.csv"
# test_dir = r"../data_list/test.csv"

train_dataset = LaneDataset(train_dir, image_size=CONFIG.IMG_SIZE,
                               offset=CONFIG.OFFSET, transform=transforms.Compose([ToTensor()]))
val_dataset = LaneDataset(val_dir, image_size=CONFIG.IMG_SIZE,
                          offset=CONFIG.OFFSET, transform=transforms.Compose([ToTensor()]))
test_dataset = LaneDataset(test_dir)


# drop_last: drop the last data in each epoches
train_loader = DataLoader(train_dataset, batch_size=CONFIG.BATCH_SIZE,
                                 shuffle=True, drop_last=True, **kwargs)

val_loader = DataLoader(val_dir, batch_size=1,
                        shuffle=False, drop_last=False, **kwargs)


