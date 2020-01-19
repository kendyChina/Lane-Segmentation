import numpy as np

"""Encode the color_mask's id to trainId"""
def encode_labels(color_mask):
    encode_mask = np.zeros((color_mask.size[0], color_mask.size[1]),
                           dtype=np.long) # size as color_mask
    # If ignoreInEval=True, then value=0
    color_mask = np.array(color_mask).T # Image(W, H) -> array(H, W) -> array(W, H)
    train_id = {0: [0, 249, 255, 213, 206, 207, 211, 208, 216, 215, 218,
                    219, 232, 202, 231, 230, 228, 229, 233, 212, 223],
                1: [200, 204, 209], 2: [201, 203], 3: [217], 4: [210],
                5: [214], 6: [220, 221, 222, 224, 225, 226], 7: [205, 227, 250]}
    for i in range(len(train_id)):
        for item in train_id[i]:
            encode_mask[color_mask == item] = i

    return encode_mask


def decode_labels(labels):
    """
    Decode the labels's trainId to id
    :param labels:
    :return:
    """
    decode_mask = np.zeros((labels.size[0], labels.size[1]), dtype='uint8')
    id = {0: 0, 1: 204, 2: 203, 3: 217, 4: 210, 5: 214, 6: 224, 7: 227}
    for i in range(len(id)):
        decode_mask[labels == i] = id[i]

    return decode_mask


def decode_color_labels(labels):
    """
    Decode the labels's trainId to color
    :param labels: numpy.ndarray
    :return:
    """
    decode_mask = np.zeros((3, labels.shape[0], labels.shape[1]), dtype='uint8')
    color_map = {0: (0, 0, 0), 1: (70, 130, 180), 2: (0, 0, 142), 3: (153, 153, 153),
                 4: (128, 64, 128), 5: (190, 153, 153), 6: (0, 0, 230), 7: (255, 128, 0)}
    for k in color_map:
        decode_mask[0][labels == k] = color_map[k][0]
        decode_mask[1][labels == k] = color_map[k][1]
        decode_mask[2][labels == k] = color_map[k][2]


    return decode_mask