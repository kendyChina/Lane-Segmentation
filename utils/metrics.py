import torch
import torch.nn.functional as F
import numpy as np
from config import CONFIG

# MIoU: Mean Intersection over Union
class MeanIoU(object):
    def getConfusionMatrix(self):
        # print(pred)
        # print(label)
        # Return True and False
        mask = (self.label >= 0) & (self.label < self.num_classes)
        # print(mask)
        # print(label[mask])
        # print(pred[mask])
        label = self.num_classes * self.label[mask] + self.pred[mask]
        # print(label)
        count = np.bincount(label, minlength=self.num_classes ** 2)
        # print(count)
        confusionMatrix = count.reshape(self.num_classes, self.num_classes)
        # print(confusionMatrix)
        self.confusionMatrix = confusionMatrix

    def meanIntersectionOverUnion(self):
        # IoU = TP / (TP + FP + FN) = intersection / union
        intersection = np.diag(self.confusionMatrix) # 取对角线值，返回list
        # print(intersection)
        union = np.sum(self.confusionMatrix, axis=1) + np.sum(self.confusionMatrix, axis=0)
        # print(union)
        IoU = []
        for i, val in enumerate(intersection):
            if union[i] != 0:
                IoU.append(intersection[i] / union[i])
            else:
                IoU.append(1.0)
        # print(IoU)
        mIoU = np.nanmean(IoU) # Compute the mean of each IoU
        # print(mIoU)
        return mIoU

    def __call__(self, pred, label):
        self.num_classes = CONFIG.NUM_CLASSES
        pred = pred.cpu()
        label = label.cpu()
        pred = torch.argmax(F.softmax(pred, dim=1), dim=1)
        # print(pred.shape)
        # print(label.shape)
        assert pred.shape[-2:] == label.shape[-2:]
        self.pred = pred
        self.label = label
        self.getConfusionMatrix()
        return self.meanIntersectionOverUnion()


def compute_iou(pred, label, iou):
    """
    pred : [N, H, W]
    label: [N, H, W]
    """
    pred = pred.cpu().numpy()
    label = label.cpu().numpy()
    for i in range(CONFIG.NUM_CLASSES):
        single_pred = pred == i
        single_label = label == i
        temp_tp = np.sum(single_pred * single_label)
        temp_ta = np.sum(single_pred) + np.sum(single_label) - temp_tp
        iou["TP"][i] += temp_tp
        iou["TA"][i] += temp_ta
    return iou