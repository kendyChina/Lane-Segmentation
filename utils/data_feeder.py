from tqdm import tqdm
import torch
from torchvision import transforms
from torch.utils.data import DataLoader
from utils.image_process import LaneDataset, ImageAug, DeformAug, ToTensor

cuda_avail = torch.cuda.is_available()
print("cuda is available: {}".format(cuda_avail))
# pin_memory: 锁页内存
kwargs = {"num_workers": 1, "pin_memory": False} if cuda_avail else {}
train_dir = r"data_list/train.csv"
val_dir = r"data_list/val.csv"
test_dir = r"data_list/test.csv"
# train_dir = r"../data_list/train.csv"
# val_dir = r"../data_list/val.csv"
# test_dir = r"../data_list/test.csv"
IMG_SIZE = [256, 96]
# IMG_SIZE = [1024, 384]
offset = 690
batch_size = 2

train_dataset = LaneDataset(train_dir, image_size=IMG_SIZE,
                               offset=offset, transform=transforms.Compose([ToTensor()]))
val_dataset = LaneDataset(val_dir, image_size=[3384, 1710],
                          offset=0, transform=transforms.Compose([ToTensor()]))
test_dataset = LaneDataset(test_dir)


# drop_last: drop the last data in each epoches
train_loader = DataLoader(train_dataset, batch_size=batch_size,
                                 shuffle=True, drop_last=True, **kwargs)
val_loader = DataLoader(val_dir, batch_size=1,
                        shuffle=True, drop_last=True, **kwargs)


