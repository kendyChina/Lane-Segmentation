import os
import pandas as pd
from sklearn.utils import shuffle

image_list = []
label_list = []

def remote_path():
    """
    sample:
    image: /root/data/LaneSeg/Image_Data/Road02/Record001/Camera 5/170927_064620420_Camera_6.jpg
    label: /root/data/LaneSeg/Gray_Label/Label_road02/Label/Record001/Camera 5170927_064620420_Camera_6_bin.png
    """
    image_dir = "/root/data/LaneSeg/Image_Data"
    label_dir = "/root/data/LaneSeg/Gray_Label"
    for s1 in os.listdir(image_dir):
        image_sub_dir1 = os.path.join(image_dir, s1) # Road2
        label_sub_dir1 = os.path.join(label_dir, "Label_{}".format(s1.lower()), "Label") # Label_road2/Label
        for s2 in os.listdir(image_sub_dir1): # Record001
            image_sub_dir3 = os.path.join(image_sub_dir1, s2)
            label_sub_dir3 = os.path.join(label_sub_dir1, s2)
            for s3 in os.listdir(image_sub_dir3): # Camera 5
                image_sub_dir4 = os.path.join(image_sub_dir3, s3)
                label_sub_dir4 = os.path.join(label_sub_dir3, s3)
                for img in os.listdir(image_sub_dir4):
                    image_sub_dir5 = os.path.join(image_sub_dir4, img)
                    label = img.replace(".jpg", "_bin.png")
                    label_sub_dir5 = os.path.join(label_sub_dir4, label)
                    if not os.path.exists(image_sub_dir5):
                        print(image_sub_dir5)
                        continue
                    if not os.path.exists(label_sub_dir5):
                        print(label_sub_dir5)
                        continue
                    image_list.append(image_sub_dir5)
                    label_list.append(label_sub_dir5)

def local_path():
    """
    sample:
    image: E:\code\cv\baidu_data_set\baidu_Image_Data\Road02\ColorImage_road02\ColorImage\Record007\Camera 6\170927_064620420_Camera_6.jpg
    label: E:\code\cv\baidu_data_set\baidu_Gray_Label\Label_road02\Label\Record007\Camera 6\170927_064620420_Camera_6_bin.png
    """
    image_dir = r"E:\code\cv\baidu_data_set\baidu_Image_Data"
    label_dir = r"E:\code\cv\baidu_data_set\baidu_Gray_Label"
    # label_dir = r"E:\code\cv\baidu_data_set\baidu_Labels_Fixed"
    for s1 in os.listdir(image_dir):
        image_sub_dir1 = os.path.join(image_dir, s1)
        label_sub_dir1 = os.path.join(label_dir, "Label_{}".format(s1.lower()))
        for s2 in os.listdir(image_sub_dir1):
            # Because image folder has one more level called ColorImage
            image_sub_dir2 = os.path.join(image_sub_dir1, s2, "ColorImage")
            label_sub_dir2 = os.path.join(label_sub_dir1, "Label")
            for s3 in os.listdir(image_sub_dir2):
                image_sub_dir3 = os.path.join(image_sub_dir2, s3)
                label_sub_dir3 = os.path.join(label_sub_dir2, s3)
                for s4 in os.listdir(image_sub_dir3):
                    image_sub_dir4 = os.path.join(image_sub_dir3, s4)
                    label_sub_dir4 = os.path.join(label_sub_dir3, s4)
                    for img in os.listdir(image_sub_dir4):
                        image_sub_dir5 = os.path.join(image_sub_dir4, img)
                        label = img.replace(".jpg", "_bin.png")
                        label_sub_dir5 = os.path.join(label_sub_dir4, label)
                        if not os.path.exists(image_sub_dir5):
                            print(image_sub_dir5)
                            continue
                        if not os.path.exists(label_sub_dir5):
                            print(label_sub_dir5)
                            continue
                        image_list.append(image_sub_dir5)
                        label_list.append(label_sub_dir5)

def run():
    remote_path()
    # local_path()
    assert len(image_list) == len(label_list)
    len_total = len(image_list)
    print("Image_list's length: {}, label_list's length: {}".format(len_total, len_total))

    total_length = len(image_list)
    sixth_length = int(total_length * 0.6)
    eighth_length = int(total_length * 0.8)

    # 0.6 for tarin, 0.2 for val, 0.2 for test
    all_dataset = pd.DataFrame({"image": image_list, "label": label_list})
    shuffle_dataset = shuffle(all_dataset)

    train_dataset = shuffle_dataset[:sixth_length]
    len_train = len(train_dataset)
    print("Train_dataset's length: {}...{:.2%} in total".format(len_train, len_train / len_total))
    val_dataset = shuffle_dataset[sixth_length: eighth_length]
    len_val = len(val_dataset)
    print("Val_dataset's length: {}...{:.2%} in total".format(len_val, len_val / len_total))
    test_dataset = shuffle_dataset[eighth_length:]
    len_test = len(test_dataset)
    print("Test_dataset's length: {}...{:.2%} in total".format(len_test, len_test / len_total))

    train_dataset.to_csv("../data_list/train.csv", index=False)
    val_dataset.to_csv("../data_list/val.csv", index=False)
    test_dataset.to_csv("../data_list/test.csv", index=False)

if __name__ == '__main__':
    run()