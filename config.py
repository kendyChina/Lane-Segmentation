import torch

class CONFIG(object):
    """
        to config the network
    """
    """network config"""
    NUM_CLASSES = 8

    """device config"""
    USE_CUDA = True
    if USE_CUDA:
        CUDA_AVAIL = torch.cuda.is_available()
    else:
        CUDA_AVAIL = False
    CUDA_DEVICE = 1

    SAVE_PATH = "logs"
    # CHECKPOINT_FILE = "checkpoint_unet.pt"
    CHECKPOINT_FILE = "checkpoint_deeplabv3p.pt"
    # LOGGING_FILE = "training_unet.csv"
    LOGGING_FILE = "training_deeplabv3p.csv"

    """train config"""
    MODEL = "deeplabv3p"
    EPOCHS = 200
    BATCH_SIZE = 4
    # IMG_SIZE = [1024, 384]
    IMG_SIZE = [256, 96]
    # IMG_SIZE = [64, 24]
    OFFSET = 690
    WEIGHT_DECAY = 1.0e-4
    BASE_LR = 6.0e-3

    """deeplabv3p config"""
    PRETRAIN = True
    OUTPUT_STRIDE = 8
    # OUTPUT_STRIDE = 16
    ASPP_OUTDIM = 256
    GROUPS_FOR_NORM = 32
    SHORTCUT_DIM = 48
    SHORTCUT_KERNEL = 1