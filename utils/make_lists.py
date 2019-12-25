import os
import pandas as pd
from sklearn.utils import shuffle

image_dir = r"E:\code\cv\baidu_data_set\baidu_Image_Data"
label_dir = r"E:\code\cv\baidu_data_set\baidu_Gray_Label"

image_list = []
label_list = []
r"""
sample:
image: E:\code\cv\baidu_data_set\baidu_Image_Data\Road02\ColorImage_road02\ColorImage\Record007\Camera 6\170927_064620420_Camera_6.jpg
label: E:\code\cv\baidu_data_set\baidu_Gray_Label\Label_Road02\Label\Record007\Camera 6\170927_064620420_Camera_6_bin.png
"""
for s1 in os.listdir(image_dir):
    image_sub_dir1 = os.path.join(image_dir, s1)
    label_sub_dir1 = os.path.join(label_dir, "Label_{}".format(s1))
    for s2 in os.listdir(image_sub_dir1):
        image_sub_dir2 = os.path.join(image_sub_dir1, s2)
        label_sub_dir2 = os.path.join(label_sub_dir1, "Label")
        for s3 in os.listdir(image_sub_dir2):
            # Because image folder has one more level called ColorImage
            image_sub_dir3 = os.path.join(image_sub_dir2, "ColorImage")
            label_sub_dir3 = label_sub_dir2
            for s4 in os.listdir(image_sub_dir3):
                image_sub_dir4 = os.path.join(image_sub_dir3, s4)
                label_sub_dir4 = os.path.join(label_sub_dir3, s4)
                for s5 in os.listdir(image_sub_dir4):
                    image_sub_dir5 = os.path.join(image_sub_dir4, s5)
                    label_sub_dir5 = os.path.join(label_sub_dir4, s5)
                    for img in os.listdir(image_sub_dir5):
                        image_sub_dir6 = os.path.join(image_sub_dir5, img)
                        label = img.replace(".jpg", "_bin.png")
                        label_sub_dir6 = os.path.join(label_sub_dir5, label)
                        if not os.path.exists(image_sub_dir6):
                            print(image_sub_dir6)
                            continue
                        if not os.path.exists(label_sub_dir6):
                            print(label_sub_dir6)
                            continue
                        image_list.append(image_sub_dir6)
                        label_list.append(label_sub_dir6)
print("image_list: {}, label_list: {}".format(len(image_list), len(label_list)))

save = pd.DataFrame({"image": image_list, "label": label_list})
save_shuffle = shuffle(save)
save_shuffle.to_csv("../data_list/train.csv", index=False)