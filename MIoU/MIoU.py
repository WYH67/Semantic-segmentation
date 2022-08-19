"""
refer to https://github.com/jfzhang95/pytorch-deeplab-xception/blob/master/utils/metrics.py
"""
import numpy as np

__all__ = ['SegmentationMetric']

"""
confusionMetric  # 注意：此处横着代表预测值，竖着代表真实值，与之前介绍的相反
P\L     P    N
P      TP    FP
N      FN    TN
"""


class SegmentationMetric(object):
    def __init__(self, numClass):
        self.numClass = numClass
        self.confusionMatrix = np.zeros((self.numClass,) * 2)

    # MIoU
    '''
    TP（True Positives）意思就是被分为了正样本，而且分对了。
    TN（True Negatives）意思就是被分为了负样本，而且分对了，
    FP（False Positives）意思就是被分为了正样本，但是分错了（事实上这个样本是负样本）。
    FN（False Negatives）意思就是被分为了负样本，但是分错了（事实上这个样本是正样本）。
    '''
    def meanIntersectionOverUnion(self):
        # Intersection = TP Union = TP + FP + FN
        # IoU = TP / (TP + FP + FN)
        intersection = np.diag(self.confusionMatrix)  # 取对角元素的值，返回列表
        union = np.sum(self.confusionMatrix, axis=1) + np.sum(self.confusionMatrix, axis=0) - np.diag(
            self.confusionMatrix)  # axis = 1表示混淆矩阵行的值，返回列表； axis = 0表示取混淆矩阵列的值，返回列表
        IoU = intersection / union  # 返回列表，其值为各个类别的IoU
        mIoU = np.nanmean(IoU)  # 求各类别IoU的平均
        return mIoU

    # 根据标签和预测图片返回其混淆矩阵
    def genConfusionMatrix(self, imgPredict, imgLabel):  # 同FCN中score.py的fast_hist()函数
        # remove classes from unlabeled pixels in gt image and predict
        mask = (imgLabel >= 0) & (imgLabel < self.numClass)
        label = self.numClass * imgLabel[mask] + imgPredict[mask]
        # np.bincount计算了从0到n**2-1这n**2个数中每个数出现的次数，返回值形状(n, n)
        count = np.bincount(label, minlength=self.numClass ** 2)
        confusionMatrix = count.reshape(self.numClass, self.numClass)
        return confusionMatrix

    # 更新混淆矩阵
    def addBatch(self, imgPredict, imgLabel):
        assert imgPredict.shape == imgLabel.shape
        self.confusionMatrix += self.genConfusionMatrix(imgPredict, imgLabel)

    # 清空混淆矩阵
    def reset(self):
        self.confusionMatrix = np.zeros((self.numClass, self.numClass))


if __name__ == '__main__':
    imgPredict = np.array([0, 0, 1, 1, 2, 2])  # 可直接换成预测图片
    imgLabel = np.array([0, 0, 1, 1, 2, 1])  # 可直接换成标注图片
    metric = SegmentationMetric(21)  # 3表示有3个分类，有几个分类就填几
    metric.addBatch(imgPredict, imgLabel)
    mIoU = metric.meanIntersectionOverUnion()
    print('mIoU is : %f' % mIoU)

