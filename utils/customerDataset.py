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
    #重新写draw_mask
    def draw_mask(self, num_obj, mask, image, image_id):
        info = self.image_info[image_id]
        for index in range(num_obj):
            for i in range(np.shape(mask)[1]):
                for j in range(np.shape(mask)[0]):
                    at_pixel = image.getpixel((i, j))
                    if at_pixel == index + 1:
                        mask[j, i, index] =1
        return mask

    #并在self.image_info信息中添加了path、mask_path 、yaml_path
    def load_shapes(self, shape_name, count, classes, img_floder, mask_floder, imglist, yaml_floder):
        for index, item in enumerate(classes):
            self.add_class(shape_name, index+1, item)
        for i in range(count):
            img = imglist[i]
            if img.endswith(".jpg"):
                img_name = img.split(".")[0]
                img_path = os.path.join(img_floder, img)
                mask_path = os.path.join(mask_floder, img_name + ".png")
                yaml_path = os.path.join(yaml_floder, img_name + ".yaml")
                self.add_image(shape_name, image_id=i, path=img_path, mask_path=mask_path,yaml_path=yaml_path)
    #重写load_mask
    def load_mask(self, image_id):
        info = self.image_info[image_id]
        img = Image.open(info['mask_path'])
        num_obj = self.get_obj_index(img)
        mask = np.zeros([np.shape(img)[0], np.shape(img)[1], num_obj], dtype=np.uint8)
        mask = self.draw_mask(num_obj, mask, img, image_id)
        labels=[]
        labels=self.from_yaml_get_class(image_id)
        # labels_form=[]
        # for i in range(len(labels)):
        #     if labels[i].find("circle")!=-1:
        #         labels_form.append("circle")
        #     elif labels[i].find("square")!=-1:
        #         labels_form.append("square")
        #     elif labels[i].find("triangle")!=-1:
        #         labels_form.append("triangle")
        class_ids = np.array([self.class_names.index(s) for s in labels])
        return mask, class_ids.astype(np.int32)