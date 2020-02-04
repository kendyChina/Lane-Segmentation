import os
from PIL import Image
import torch
import numpy as np
import pandas as pd
from sklearn.utils import shuffle
from config import CONFIG
from models.deeplabv3p_resnet import RESNETDeeplabV3Plus
from models.UNet import UNet
from utils.image_process import crop_resize_data, reshape_data
from utils.process_labels import decode_color_labels
from utils.visualize import PlotImage



#os.environ["CUDA_VISIBLE_DEVICES"] = ""
#for dvi in range(torch.cuda.device_count()):
#    print(torch.cuda.get_device_name(dvi))


nets = {'deeplabv3p': RESNETDeeplabV3Plus, 'unet': UNet}
cuda_device = 5
offset = CONFIG.OFFSET


def load_model(predict_net, model_path):

    net = nets[predict_net](pretrained=False, batch_norm=True, bias=False)
    net.eval()
    if torch.cuda.is_available():
        torch.cuda.set_device(cuda_device)
        net = net.cuda()
        map_location = "cuda:{}".format(cuda_device)
    else:
        map_location = 'cpu'

    model_param = torch.load(model_path, map_location=map_location)
    model_param = {k.replace('module.', ''):v for k, v in model_param.items()}
    net.load_state_dict(model_param)
    return net


def img_transform(img):
    img = crop_resize_data(img, image_size=[1024, 384], offset=offset)
    img = np.transpose(img, (2, 0, 1))
    img = img[np.newaxis, ...].astype(np.float32)
    img = torch.from_numpy(img.copy())
    if torch.cuda.is_available():
        img = img.cuda()
    return img


def resize_decode_color(pred, image_size):
    pred = torch.softmax(pred, dim=1)
    pred_heatmap = torch.max(pred, dim=1)
    # 1,H,W,C
    pred = torch.argmax(pred, dim=1)
    pred = torch.squeeze(pred)
    pred = pred.detach().cpu().numpy()
    pred = np.uint8(pred)
    pred = reshape_data(pred, image_size, offset=offset)
    pred = decode_color_labels(pred)
    pred = np.transpose(pred, (1, 2, 0))
    return pred


def main():
    predict_net = 'unet'
    model_dir = 'logs'
    model_path = os.path.join(model_dir,  'unet_downchnl_weight.pt')
    net = load_model(predict_net, model_path)

    plot_img = PlotImage("image", "predict")
    plot_pred = PlotImage("pred", "predict")
    plot_label = PlotImage("label", "predict")

    test_loader = shuffle(pd.read_csv("data_list/val.csv"))
    image = test_loader["image"]
    label = test_loader["label"]

    img = Image.open(image[0])
    plot_img(img)
    W, H = img.size  # original size
    img = img_transform(img)

    pred = net(img)
    color_mask = resize_decode_color(pred, (W, H))
    color_mask = Image.fromarray(color_mask).convert("RGB")
    plot_pred(color_mask)
    plot_label(label[0])

    color_mask.save(os.path.join('color_mask.png'))


if __name__ == '__main__':
    main()
    # test_loader = shuffle(pd.read_csv("data_list/val.csv"))
    # image = test_loader["image"]
    # label = test_loader["label"]
    # print(image[0])

