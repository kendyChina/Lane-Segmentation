import numpy as np

from tqdm import tqdm
from config import CONFIG
from utils.data_feeder import train_loader, val_loader, test_loader


def main(loader):
    num_classes = {i: 0 for i in range(CONFIG.NUM_CLASSES)}
    for item in tqdm(loader):
        label = item["label"].numpy()
        for i in range(CONFIG.NUM_CLASSES):
            num_i = np.sum(label == i)
            num_classes[i] += num_i
    total_pixels = sum(num_classes.values())
    for i in range(CONFIG.NUM_CLASSES):
        print("Class{} has number of {}...{:.2} in total.".format(i, num_classes[i], num_classes[i] / total_pixels))

if __name__ == '__main__':
    print("train_loader:")
    main(train_loader)
    print("val_loader:")
    main(val_loader)
    print("test_loader")
    main(test_loader)