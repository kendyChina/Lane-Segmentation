import numpy as np
import pandas as pd
from PIL import Image
from imgaug import augmenters as iaa
import torch
from torch.utils.data import Dataset
from utils.process_labels import encode_labels


def crop_resize_data(image, label=None, image_size=[1024, 384], offset=690):
    """
    Crop and resize the data or label
    :param image: PIL.Image
    :param label: PIL.Image
    :param image_size: list or tuple with two values
    :param offset: int
    :return:
    """
    (width, height) = image.size
    (left, upper, right, lower) = (0, offset, width, height)
    roi_image = image.crop((left, upper, right, lower))
    # im.resize((width, height), resample=mode)
    train_image = roi_image.resize(tuple(image_size), resample=Image.BILINEAR)

    if label is not None:
        roi_label = label.crop((left, upper, right, lower))
        train_label = roi_label.resize(tuple(image_size), resample=Image.NEAREST)
        return train_image, train_label

    return train_image


def reshape_data(pred, image_size, offset=0):
    W, H = image_size
    h = H - offset
    pred = Image.fromarray(pred)
    pred = pred.resize((W, h), resample=Image.BILINEAR)
    pred = np.array(pred)
    pred = np.pad(pred, ((offset, 0), (0, 0)))
    return pred


class LaneDataset(Dataset):
    def __init__(self, csv_file, image_size=[1024, 384], offset=690, transform=None):
        super(LaneDataset, self).__init__()
        data = pd.read_csv(csv_file)
        self.image = data["image"].values
        self.label = data["label"].values
        assert self.image.shape[0] == self.label.shape[0]

        self.image_size = image_size
        self.offset = offset
        self.transform = transform

    def __len__(self):
        return self.image.shape[0]

    def __getitem__(self, idx):
        ori_image = Image.open(self.image[idx])
        ori_label = Image.open(self.label[idx])
        ori_image, ori_label = crop_resize_data(ori_image, ori_label,
                                                self.image_size, self.offset)
        # Image -> ndarray (W, H)
        ori_image = np.array(ori_image)
        # ori_image = ori_image
        ori_label = encode_labels(ori_label)

        sample = [ori_image.copy(), ori_label.copy()]
        if self.transform:
            sample = self.transform(sample)
        return sample

"""Pixel augmentation"""
class ImageAug(object):
    def __call__(self, sample):
        image, label = sample
        if np.random.uniform(0, 1) >= 0.5:
            seq = iaa.Sequential([iaa.OneOf([
                iaa.AdditiveGaussianNoise(scale=(0, 0.2 * 255)),
                iaa.Sharpen(alpha=(0.1, 0.3), lightness=(0.7, 1.3)),
                iaa.GaussianBlur(0.0, 1.0),])])
            image = seq.augment_image(image)
        return image, label

"""Deformation augmentation"""
class DeformAug(object):
    def __call__(self, sample):
        image, label = sample
        seq = iaa.Sequential([iaa.CropAndPad(percent=(-0.05, 0.1))])
        seq_to = seq.to_deterministic()
        image = seq_to.augment_image(image)
        label = seq_to.augment_image(label)
        return image, label

class ToTensor(object):
    def __call__(self, sample):
        image, label = sample
        # Transpose (H, W, C) to (C, H, W)
        image = image.transpose((2, 0, 1))
        # Transpose (W, H) to (H, W)
        label = label.transpose((1, 0))
        image = image.astype(np.float32)
        label = label.astype(np.long)
        return {"image": torch.from_numpy(image.copy()),
                "label": torch.from_numpy(label.copy()).long()}

if __name__ == '__main__':
    # img = Image.open(r"E:\code\cv\baidu_data_set\baidu_Image_Data\Road02\ColorImage_road02\ColorImage\Record001\Camera 5\170927_063811892_Camera_5.jpg")
    # crop_resize_data(img, img)
    pass