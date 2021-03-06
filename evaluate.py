import cv2
import tensorflow as tf
import numpy as np
import config
import os
from mrcnn.mask_rcnn import MASK_RCNN
from PIL import Image

'''
图像分割评价指标：
mIoU：类别平均交并比
PixelAccuracy：像素精度，标记正确的像素占总像素的比例
Mean Pixel accuracy：平均像素精度，每个类内被正确分类像素数的比例
参考：https://zhuanlan.zhihu.com/p/61880018 
'''

def IoU_calculate(pred, target, n_classes):
    ious = []
    # ignore IOU for background class
    for item in range(1, n_classes):
        pred_inds =pred==item
        target_inds = target==item
        intersection = (pred_inds[target_inds]).sum()
        union = pred_inds.sum()+target_inds.sum()-intersection
        if union==0:
            # if there is no ground true, do not include in evaluation
            ious.append(float('nan'))
        else:
            ious.append(float(intersection)/float(max(union, 1)))
    return ious

# numpy版本

def all_iou(a, b, n):
    '''
    a: ground true, shape:h*w
    b: prediction, shape: h*w
    n: class
    '''
    # 找出ground true中需要的类别
    k = (a>0)&(a<=n)
    return np.bincount(n*a[k].astype(int)+b[k], minlength=n**2).reshape(n, n)

def per_class_iou(hist):
    '''
    分别为每个类别计算mIoU
    '''
    # 矩阵的对角线上的值组成的一维数组/矩阵的所有元素之和
    return np.diag(hist)/(hist.sum(1)+hist.sum(0)-np.diag(hist))

def mIoU_metric(pred, target, n_classes):
    hist = np.zeros((n_classes, n_classes))
    # 对图像进行计算hist矩阵并累加
    hist+= all_iou(target.flattern(), pred.flattern(), n_classes)
    # 计算每个类别的iou
    mIoUs = per_class_iou(hist)
    for ind_class in range(n_classes):
        print(str(round(mIoUs[ind_class]*100, 2)))
    print('--->mIoU：'+str(round(np.nanmean(mIoUs)*100, 2)))
    return mIoUs


class Evaluator(object):
    def __init__(self, num_class) -> None:
        super().__init__()
        self.num_class = num_class
        self.confusion_matrix = np.zeros((self.num_class, )*2)
    
    def Pixel_Accuracy(self):
        Acc = np.diag(self.confusion_matrix).sum()/self.confusion_matrix.sum()
        return Acc
    
    def Pixel_Accuracy_Class(self):
        Acc = np.diag(self.confusion_matrix)/self.confusion_matrix.sum(axis=1)
        Acc = np.nanmean(Acc)
        return Acc
    
    def Mean_Intersection_over_Union(self):
        MIoU = np.diag(self.confusion_matrix) / (
                    np.sum(self.confusion_matrix, axis=1) + np.sum(self.confusion_matrix, axis=0) -
                    np.diag(self.confusion_matrix))
        MIoU = np.nanmean(MIoU)
        return MIoU

    def Frequency_Weighted_Intersection_over_Union(self):
        freq = np.sum(self.confusion_matrix, axis=1) / np.sum(self.confusion_matrix)
        iu = np.diag(self.confusion_matrix) / (
                    np.sum(self.confusion_matrix, axis=1) + np.sum(self.confusion_matrix, axis=0) -
                    np.diag(self.confusion_matrix))

        FWIoU = (freq[freq > 0] * iu[freq > 0]).sum()
        return FWIoU

    def _generate_matrix(self, gt_image, pre_image):
        mask = (gt_image >= 0) & (gt_image < self.num_class)
        label = self.num_class * gt_image[mask].astype('int') + pre_image[mask]
        count = np.bincount(label, minlength=self.num_class**2)
        confusion_matrix = count.reshape(self.num_class, self.num_class)
        return confusion_matrix

    def add_batch(self, gt_image, pre_image):
        '''
        输入的图像用0,1,2,3...表示类别
        '''
        assert gt_image.shape == pre_image.shape
        self.confusion_matrix += self._generate_matrix(gt_image, pre_image)

    def reset(self):
        self.confusion_matrix = np.zeros((self.num_class,) * 2)



if __name__ == '__main__':
    ori_img = './train_data/20211229095227.jpg'
    gt_img = './train_data/20211229095227.png'
    pred_img = './train_data/save.png'
    image = Image.open(ori_img)
    gt_img = Image.open(gt_img)
    pred_img = Image.open(pred_img)
    pred_img = np.where(pred_img, 0, 1)
    gt_img = np.array(gt_img)
    gt_img = np.where(gt_img>0, 0, 1)
    pred_img = np.array(pred_img)
    mask_rcnn = MASK_RCNN()
    class_names = mask_rcnn.get_class()
    n_classes = len(class_names)
    result_img, pred_img = mask_rcnn.detect_image(image=image)
    result_img.show()
    evaluate = Evaluator(2)
    evaluate.add_batch(gt_img, pred_img)
    acc = evaluate.Pixel_Accuracy()
    print(acc)


    # iou计算
    iou = IoU_calculate(pred_img, gt_img, 1)
    print(iou)




