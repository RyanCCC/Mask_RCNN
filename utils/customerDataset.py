import os
import numpy as np
from PIL import Image
import yaml
from .dataset import Dataset
from .utils import non_max_suppression

class CustomerDataset(Dataset):
    #得到该图中有多少个实例（物体）
    def get_obj_index(self, image):
        n = np.max(image)
        return n
    #解析labelme中得到的yaml文件，从而得到mask每一层对应的实例标签
    def from_yaml_get_class(self,image_id):
        info=self.image_info[image_id]
        with open(info['yaml_path']) as f:
            temp=yaml.load(f.read(), Loader=yaml.FullLoader)
            labels=temp['label_names']
            del labels[0]
        return labels

    def draw_mask(self, num_obj, mask, image, image_id):
        info = self.image_info[image_id]
        for index in range(num_obj):
            for i in range(np.shape(mask)[1]):
                for j in range(np.shape(mask)[0]):
                    at_pixel = image.getpixel((i, j))
                    if at_pixel == index + 1:
                        mask[j, i, index] =1 
        save_path = os.path.dirname(info['mask_path'])
        base_name = os.path.basename(info['path'])
        base_name = os.path.splitext(base_name)[0]
        np.savez_compressed(os.path.join(save_path, base_name), mask)
        return mask

    #并在self.image_info信息中添加了path、mask_path 、yaml_path
    def load_shapes(self, shape_name, count, classes, img_floder, mask_floder, imglist, yaml_floder, train_mode = True):
        for index, item in enumerate(classes):
            self.add_class(shape_name, index+1, item)
        for i in range(count):
            img = imglist[i]
            if img.endswith(".jpg"):
                img_name = img.split(".")[0]
                img_path = os.path.join(img_floder, img)
                if train_mode:
                    mask_path = os.path.join(mask_floder, img_name + ".npz")
                else:
                    # npz文件加载
                    mask_path = os.path.join(mask_floder, img_name + ".png")
                yaml_path = os.path.join(yaml_floder, img_name + ".yaml")
                self.add_image(shape_name, image_id=i, path=img_path, mask_path=mask_path,yaml_path=yaml_path)
    #重写load_mask
    def load_mask(self, image_id, train_mode = True):
        info = self.image_info[image_id]
        if train_mode:
            # 训练模式下加载npz数据
            mask = np.load(info['mask_path'])['arr_0']
        else:
            # 生成npz文件
            img = Image.open(info['mask_path'])
            num_obj = self.get_obj_index(img)
            mask = np.zeros([np.shape(img)[0], np.shape(img)[1], num_obj], dtype=np.uint8)
            mask = self.draw_mask(num_obj, mask, img, image_id)
        labels=[]
        labels=self.from_yaml_get_class(image_id)
        class_ids = np.array([self.class_names.index(s) for s in labels])
        return mask, class_ids.astype(np.int32)