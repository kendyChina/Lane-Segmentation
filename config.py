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
    SET_DEVICE = 3

    SAVE_PATH = "logs"

    # train config
    EPOCHS = 200
    SAVE_EPOCH = 2
    BATCH_SIZE = 1
    # IMG_SIZE = [1024, 384]
    IMG_SIZE = [256, 96]
    OFFSET = 690
    WEIGHT_DECAY = 1.0e-4
    BASE_LR = 6.0e-4