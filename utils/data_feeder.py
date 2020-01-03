from tqdm import tqdm
import torch
from torchvision import transforms
from torch.utils.data import DataLoader
from utils.image_process import LaneDataset, ImageAug, DeformAug, ToTensor

cuda_avail = torch.cuda.is_available()
print("cuda is available: {}".format(cuda_avail))
# pin_memory: 锁页内存
kwargs = {"num_workers": 1, "pin_memory": False} if cuda_avail else {}
data_dir = r"data_list/train.csv"
IMG_SIZE = [256, 96]
offset = 690
batch_size = 1
training_dataset = LaneDataset(data_dir, image_size=IMG_SIZE,
                               offset=offset, transform=transforms.Compose([ToTensor()]))

# drop_last: drop the last data in each epoches
training_data_batch = DataLoader(training_dataset, batch_size=batch_size,
                                 shuffle=True, drop_last=True, **kwargs)

# dataprocess = tqdm(training_data_batch)
dataprocess = training_data_batch
