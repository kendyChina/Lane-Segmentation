from PIL import Image
import numpy as np
import os
from utils.image_process import crop_resize_data
from utils.process_labels import encode_labels

def train_image_gen(train_list, batch_size=4, image_size=[1024, 384], crop_offset=690):
    all_batches_index = np.arange(len(train_list))
    out_images = []
    out_labels = []
    image_dir = np.array(train_list["image"])
    label_dir = np.array(train_list["label"])
    while True:
        """Shuffle the indexes each epoch"""
        np.random.shuffle(all_batches_index)
        for index in all_batches_index:
            if os.path.exists(image_dir[index]) and os.path.exists(label_dir[index]):
                # original
                ori_image = Image.open(image_dir[index])
                ori_label = Image.open(label_dir[index]).convert("L")
                """
                Crop the top part of image
                Resize the train image
                """
                ori_image, ori_label = crop_resize_data(ori_image, ori_label, image_size,
                                                        crop_offset)
                ori_label = encode_labels(ori_label)
                # Just string, bytes-like object or a number can be arrayed
                out_images.append(np.array(ori_image)) # Image to ndarray
                out_labels.append(ori_label) # ndarray
                if len(out_images) >= batch_size:
                    out_images = np.array(out_images)
                    out_labels = np.array(out_labels)
                    # print(out_images.shape) # (4, 384, 1024, 3)
                    # print(out_labels.shape) # (4, 1024, 384)
                    # print(out_images[0].shape) # (384, 1024, 3)
                    # print(out_labels[0].shape) # (1024, 384)
                    # img = Image.fromarray(out_images[0])
                    # print(img.size) # (1024, 384)
                    # img.show()
                    # out_images = out_images[:, :, :, ::-1].transpose(0, 3, 1, 2).astype(np.float32) / (255.0 / 2.0) - 1
                    """Normalization to -1 ~ 1"""
                    # img(0~255) -> img(0~2) -> img(-1~1)
                    out_images = out_images.transpose(0, 3, 1, 2).astype(np.float32) / (255.0 / 2.0) - 1
                    out_labels = out_labels.astype(np.int64)
                    yield out_images, out_labels
                    out_images = []
                    out_labels = []
            else: # If image or label is not exist
                print(image_dir[index])
                print(label_dir[index])


if __name__ == '__main__':
    import pandas as pd
    train_list = pd.read_csv(r"E:\code\cv\Lane Segmentation\data_list\train.csv")
    train_image_gen(train_list)