import numpy as np

"""Encode the color_mask's id to trainId"""
def encode_labels(color_mask):
    encode_mask = np.zeros((color_mask.size[0], color_mask.size[1]),
                           dtype=np.int64) # size as color_mask
    # If ignoreInEval=True, then value=0
    train_id = {0: [0, 249, 255, 213, 206, 207, 211, 208, 216, 215, 218,
                    219, 232, 202, 231, 230, 228, 229, 233, 212, 223],
                1: [200, 204, 209], 2: [201, 203], 3: [217], 4: [210],
                5: [214], 6: [220, 221, 222, 224, 225, 226], 7: [205, 227, 250]}
    for i in range(len(train_id)):
        for item in train_id[i]:
            encode_mask[color_mask == item] = i

    return encode_mask

"""Decode the labels's trainId to id"""
def decode_labels(labels):
    decode_mask = np.zeros((labels.size[0], labels.size[1]), dtype='uint8')
    id = {0: 0, 1: 204, 2: 203, 3: 217, 4: 210, 5: 214, 6: 224, 7: 227}
    for i in range(len(id)):
        decode_mask[labels == i] = id[i]

    return decode_mask

"""Decode the labels's trainId to color"""
def decode_color_labels(labels):
    decode_mask = np.zeros((3, labels.size[0], labels.size[1]), dtype='uint8')
    # 0
    decode_mask[0][labels == 0] = 0
    decode_mask[1][labels == 0] = 0
    decode_mask[2][labels == 0] = 0
    # 1
    decode_mask[0][labels == 1] = 70
    decode_mask[1][labels == 1] = 130
    decode_mask[2][labels == 1] = 180
    # 2
    decode_mask[0][labels == 2] = 0
    decode_mask[1][labels == 2] = 0
    decode_mask[2][labels == 2] = 142
    # 3
    decode_mask[0][labels == 3] = 153
    decode_mask[1][labels == 3] = 153
    decode_mask[2][labels == 3] = 153
    # 4
    decode_mask[0][labels == 4] = 128
    decode_mask[1][labels == 4] = 64
    decode_mask[2][labels == 4] = 128
    # 5
    decode_mask[0][labels == 5] = 190
    decode_mask[1][labels == 5] = 153
    decode_mask[2][labels == 5] = 153
    # 6
    decode_mask[0][labels == 6] = 0
    decode_mask[1][labels == 6] = 0
    decode_mask[2][labels == 6] = 230
    # 7
    decode_mask[0][labels == 7] = 255
    decode_mask[1][labels == 7] = 128
    decode_mask[2][labels == 7] = 0

    return decode_mask