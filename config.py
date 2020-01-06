import torch

class CONFIG(object):
    # model config
    NUM_CLASSES = 8

    # train config
    USE_CUDA = True
    if USE_CUDA:
        CUDA_AVAIL = torch.cuda.is_available()
    else:
        CUDA_AVAIL = False
    SAVE_PATH = "logs"
    SET_DEVICE = 2
    EPOCHS = 200
    BATCH_SIZE = 4
    # IMG_SIZE = [1024, 384]
    IMG_SIZE = [256, 96]
    OFFSET = 690
    WEIGHT_DECAY = 1.0e-4
    SAVE_PATH = "logs"
    BASE_LR = 6.0e-4