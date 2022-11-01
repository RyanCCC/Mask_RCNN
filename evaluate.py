import cv2
import tensorflow as tf
import numpy as np
import config
import os
from mrcnn.mask_rcnn import MASK_RCNN
from PIL import Image
from utils import utils, dataset, visualize
from mrcnn.mrcnn_training import load_image_gt
import yaml
import matplotlib.pyplot as plt
from tqdm import tqdm

mask_rcnn = MASK_RCNN(model=config.InferenceConfig.model, classes_path = config.InferenceConfig.class_path)
class_names = mask_rcnn.get_class()

'''
参考：
1. https://github.com/matterport/Mask_RCNN/blob/master/mrcnn/utils.py#L715
2. https://github.com/matterport/Mask_RCNN/issues/2513
3. https://zhuanlan.zhihu.com/p/61880018 
'''

class Evaluator(object):
    def __init__(self, num_class) -> None:
        super().__init__()
        self.num_class = num_class
        self.confusion_matrix = np.zeros((self.num_class, )*2)
    
    def Pixel_Accuracy(self):
        Acc = np.diag(self.confusion_matrix).sum()/self.confusion_matrix.sum()
        return Acc
    
    def Pixel_Recall(self, class_index):
        Acc = self.confusion_matrix[class_index][class_index]/self.confusion_matrix.sum(axis=0)[class_index]
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

class TestDataset(dataset.Dataset):
    # 获取图中的实例个数
    def get_obj_index(self, image):
        n = np.max(image)
        return n
    
    def get_class(self):
        classes_path = os.path.expanduser(self.classes_path)
        with open(classes_path) as f:
            class_names = f.readlines()
        class_names = [c.strip() for c in class_names]
        class_names.insert(0,"BG")
        return class_names

    # 解析yaml
    def get_classes_from_yaml(self, image_id):
        info = self.image_info[image_id]
        with open(info['yaml_path']) as f:
            temp = yaml.load(f.read())
            labels = temp['label_names']
            del labels[0]
        return labels
    
    def draw_mask(self, num_obj, mask, image, image_id):
        info = self.image_info[image_id]
        for index in range(num_obj):
            for i in range(info['width']):
                for j in range(info['height']):
                    at_pixel = image.getpixel((i, j))
                    if at_pixel == index + 1:
                        mask[j, i, index] = 1
        return mask
    
    def load_dataset(self, count, img_floder, mask_floder, imglist, dataset_root_path):
        """
        Generate the requested number of synthetic images.
        count: number of images to generate.
        height, width: the size of the generated images.
        """
        # Add classes
        classes_names = config.get_class(config.InferenceConfig.class_path)
        for index, item in enumerate(classes_names):
            self.add_class('TestSet', index+1, item)

        for i in range(count):
            # 获取图片宽和高
            filestr = imglist[i].split(".")[0]
            mask_path = mask_floder + "/" + filestr + ".png"
            yaml_path = dataset_root_path + "/" +"yaml/" + filestr + ".yaml"
            print(dataset_root_path + "labelme_json/" + filestr + "_json/img.png")
            cv_img = cv2.imread(dataset_root_path + "/" +"imgs/" + filestr + ".jpg")

            self.add_image("TestSet", image_id=i, path=img_floder + "/" + imglist[i],
                           width=cv_img.shape[1], height=cv_img.shape[0], mask_path=mask_path, yaml_path=yaml_path)

    def load_mask(self, image_id):
        """Generate instance masks for shapes of the given image ID.
        """
        global iter_num
        print("image_id", image_id)
        info = self.image_info[image_id]
        count = 1  # number of object
        img = Image.open(info['mask_path'])
        num_obj = self.get_obj_index(img)
        mask = np.zeros([info['height'], info['width'], num_obj], dtype=np.uint8)
        mask = self.draw_mask(num_obj, mask, img, image_id)
        occlusion = np.logical_not(mask[:, :, -1]).astype(np.uint8)
        for i in range(count - 2, -1, -1):
            mask[:, :, i] = mask[:, :, i] * occlusion

            occlusion = np.logical_and(occlusion, np.logical_not(mask[:, :, i]))
        labels = []
        labels = self.get_classes_from_yaml(image_id)
        class_ids = np.array([self.class_names.index(s) for s in labels])
        return mask, class_ids.astype(np.int32)

def text_save(filename, data):
    file = open(filename,'a')
    for i in range(len(data)):
        s = str(data[i]).replace('[','').replace(']','')
        s = s.replace("'",'').replace(',','') +'\n'   
        file.write(s)
    file.close()
    print(f'save success:{filename}')

if __name__ == '__main__':
    dataset_root_path = config.CustomerConfig.TRAIN_DATASET
    img_floder =os.path.join(dataset_root_path, "imgs")
    mask_floder = os.path.join(dataset_root_path, "mask")
    imglist = os.listdir(img_floder)
    count = len(imglist)
    np.random.seed(10101)
    np.random.shuffle(imglist)
    train_imglist = imglist[:int(count*0.8)]
    test_imglist = imglist[int(count*0.8):]
    test_count = len(test_imglist)

    # 加载测试集
    dataset_test = TestDataset()
    dataset_test.load_dataset(test_count, img_floder, mask_floder, test_imglist, dataset_root_path)
    dataset_test.prepare()
    APs = []
    flag = 0
    for imageid in tqdm(dataset_test.image_ids[:20]):
        image, image_meta, gt_class_id, gt_bbox, gt_mask = \
            load_image_gt(dataset_test, config.InferenceConfig, imageid)
        # 将所有ground truth载入并保存
        if flag == 0:
            gt_boxes, gt_class_ids, gt_masks = gt_bbox, gt_class_id, gt_mask
        else:
            gt_boxes = np.concatenate((gt_boxes, gt_bbox), axis=0)
            gt_class_ids = np.concatenate((gt_class_ids, gt_class_id), axis=0)
            gt_masks = np.concatenate((gt_masks, gt_mask), axis=2)
        image = Image.fromarray(image)
        r = mask_rcnn.get_detections(image=image)
        if flag == 0:
            pred_rois, pred_ids, pred_scores, pred_masks = r["rois"], r["class_ids"], r["scores"],  r['masks']
        else:
            pred_rois = np.concatenate((pred_rois, r["rois"]), axis=0)
            pred_ids = np.concatenate((pred_ids, r["class_ids"]), axis=0)
            pred_scores = np.concatenate((pred_scores, r["scores"]), axis=0)
            pred_masks = np.concatenate((pred_masks, r['masks']), axis=2)
        flag+=1
        # 展示数据
        drawed_image = visualize.display_instances(image, r['rois'], r['masks'], r['class_ids'], mask_rcnn.class_names, r['scores'], show_bbox=False, captions=False)
        # 处理mask 文件
        mask_image = np.any(r['masks'], axis=-1)
        mask_image = Image.fromarray(mask_image)
        drawed_image.show()
        mask_image.show()

    iou_thresholds = [0.5, 0.6, 0.7, 0.8, 0.9]
    # AP, precisions, recalls, overlaps =utils.compute_ap(gt_bbox, gt_class_id, gt_mask,r["rois"], r["class_ids"], r["scores"], r['masks'], iou_threshold=iou_threshold)
    # 计算AP, precision, recall
    for iou_threshold in iou_thresholds:
        AP, precisions, recalls, overlaps = utils.compute_ap(gt_boxes, gt_class_ids, gt_masks, pred_rois, pred_ids, pred_scores, pred_masks, iou_threshold=iou_threshold)
        print(f'AP@{iou_threshold}：{AP}')
        print(f"mAP@{iou_threshold}: ", np.mean(AP))
        # 保存precision, recall信息用于后续绘制图像
    #     text_save(f'Kpreci@{iou_threshold}.txt', precisions)
    #     text_save(f'Krecall@{iou_threshold}.txt', recalls)
    #     text_save(f'KAP@{iou_threshold}.txt', [AP])
    # plt.plot(recalls, precisions, 'b', label='PR')
    # plt.title('precision-recall curve')
    # plt.xlabel('Recall')
    # plt.ylabel('Precision')
    # plt.legend()
    # plt.show()
    






        # '''
        # Pixel Accuracy
        # '''
        # basename = os.path.splitext(imageid)[0]
        # ori_img = os.path.join(img_floder, imageid)
        # gt_img = os.path.join(mask_floder, basename+'.png')
        # image = Image.open(ori_img)
        # gt_img = Image.open(gt_img)
        # n_classes = len(class_names)
        # result_img, pred_img = mask_rcnn.detect_image(image=image)
        # pred_img.show()
        # gt_img.show()
        # evaluate = Evaluator(1+1)
        # evaluate.add_batch(np.array(gt_img), np.array(pred_img))
        # acc = evaluate.Pixel_Accuracy()
        # print('ACC:',acc)
        # recall = evaluate.Pixel_Recall(0)
        # print('Recall:', recall)
        # basename = os.path.splitext(imageid)[0]
        # image.save(os.path.join('./result', 'ori_'+basename+'.jpg'))
        # pred_img.save(os.path.join('./result', 'res_'+basename+'.jpg'))
        # iou计算: TODO:FIXBUG
        # iou = IoU_calculate(pred_img, gt_img, 2)
        # print(iou)




