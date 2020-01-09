import torch

class CONFIG(object):
    # model config
    NUM_CLASSES = 8

    # device config
    USE_CUDA = True
    if USE_CUDA:
        CUDA_AVAIL = torch.cuda.is_available()
    else:
        CUDA_AVAIL = False
    SET_DEVICE = 2

    SAVE_PATH = "logs"

    # train config
    EPOCHS = 200
    BATCH_SIZE = 2
    # IMG_SIZE = [1024, 384]
    IMG_SIZE = [256, 96]
    # IMG_SIZE = [64, 24]
    OFFSET = 690
    WEIGHT_DECAY = 1.0e-4
    BASE_LR = 6.0e-4