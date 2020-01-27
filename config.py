import torch
import numpy as np

class CONFIG(object):
    """
        to config the project
    """
    """network config"""
    NUM_CLASSES = 8

    """device config"""
    USE_CUDA = True
    if USE_CUDA:
        CUDA_AVAIL = torch.cuda.is_available()
    else:
        CUDA_AVAIL = False
    CUDA_DEVICE = 3

    SAVE_PATH = "logs"
    # CHECKPOINT_FILE = "unet_weight.pt"
    # CHECKPOINT_FILE = "deeplabv3p_resnet50_weight.pt"
    CHECKPOINT_FILE = "deeplabv3p_mobilenet_weight.pt"
    # LOGGING_FILE = "unet_logs.csv"
    # LOGGING_FILE = "deeplabv3p_resnet50_logs.csv"
    LOGGING_FILE = "deeplabv3p_mobilenet_logs.csv"

    """train config"""
    # MODEL = "unet"
    # MODEL = "deeplabv3p_resnet"
    MODEL = "deeplabv3p_mobilenet"
    EPOCHS = 200
    BATCH_SIZE = 4
    # IMG_SIZE = [1024, 384]
    IMG_SIZE = [256, 96]
    # IMG_SIZE = [64, 24]
    MIN_LOSS = np.Inf
    # MIN_LOSS = 0.039740
    OFFSET = 690
    WEIGHT_DECAY = 1.0e-4
    BASE_LR = 1e-3

    """deeplabv3p config"""
    PRETRAIN = True
    OUTPUT_STRIDE = 8
    # OUTPUT_STRIDE = 16
    ASPP_OUTDIM = 256
    GROUPS_FOR_NORM = 32
    SHORTCUT_DIM = 48
    SHORTCUT_KERNEL = 1