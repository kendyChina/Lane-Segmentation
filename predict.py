import os
from PIL import Image
import torch
import numpy as np
from config import CONFIG
from models.deeplabv3p import DeeplabV3Plus
from models.UNet import UNet
from utils.image_process import crop_resize_data, reshape_data
from utils.process_labels import decode_color_labels

#os.environ["CUDA_VISIBLE_DEVICES"] = ""
#for dvi in range(torch.cuda.device_count()):
#    print(torch.cuda.get_device_name(dvi))

nets = {'deeplabv3p': DeeplabV3Plus, 'unet': UNet}


def load_model(predict_net, model_path):

    net = nets[predict_net](pretrained=False)
    net.eval()
    if CONFIG.CUDA_AVAIL:
        torch.cuda.set_device(CONFIG.CUDA_DEVICE)
        net = net.cuda()
        map_location = "cuda:{}".format(CONFIG.CUDA_DEVICE)
    else:
        map_location = 'cpu'

    model_param = torch.load(model_path, map_location=map_location)
    model_param = {k.replace('module.', ''):v for k, v in model_param.items()}
    net.load_state_dict(model_param)
    return net


def img_transform(img):
    img = crop_resize_data(img, image_size=CONFIG.IMG_SIZE, offset=CONFIG.OFFSET)
    img = np.transpose(img, (2, 0, 1))
    img = img[np.newaxis, ...].astype(np.float32)
    img = torch.from_numpy(img.copy())
    if CONFIG.CUDA_AVAIL:
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
    pred = reshape_data(pred, image_size, offset=CONFIG.OFFSET)
    pred = decode_color_labels(pred)
    pred = np.transpose(pred, (1, 2, 0))
    return pred


def main():
    predict_net = 'unet'
    model_dir = 'logs'
    test_dir = r'/root/data/LaneSeg/Image_Data/Road02/Record001/Camera 5/'
    model_path = os.path.join(model_dir,  'checkpoint_unet.pt')
    net = load_model(predict_net, model_path)

    img_path = os.path.join(test_dir, '170927_063811892_Camera_5.jpg')
    img = Image.open(img_path)
    W, H = img.size # original size
    img = img_transform(img)

    pred = net(img)
    color_mask = resize_decode_color(pred, (W, H))
    color_mask = Image.fromarray(color_mask).convert("RGB")

    color_mask.save(os.path.join('color_mask.png'))


if __name__ == '__main__':
    main()

