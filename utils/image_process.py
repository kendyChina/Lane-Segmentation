from PIL import Image

def crop_resize_data(image, label=None, image_size=[1024, 384], offset=690):
    (width, height) = image.size
    (left, upper, right, lower) = (0, offset, width, height)
    roi_image = image.crop((left, upper, right, lower))
    if label is not None:
        roi_label = label.crop((left, upper, right, lower))
        # im.resize((width, height), resample=mode)
        train_image = roi_image.resize(tuple(image_size), resample=Image.BILINEAR)
        train_label = roi_label.resize(tuple(image_size), resample=Image.NEAREST)
        return train_image, train_label
    else:
        train_image = roi_image.resize(tuple(image_size), resample=Image.BILINEAR)
        return train_image

if __name__ == '__main__':
    img = Image.open(r"E:\code\cv\baidu_data_set\baidu_Image_Data\Road02\ColorImage_road02\ColorImage\Record001\Camera 5\170927_063811892_Camera_5.jpg")
    crop_resize_data(img, img)