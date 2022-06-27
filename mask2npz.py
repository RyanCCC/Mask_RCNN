from config import CustomerConfig
import os
from tqdm import tqdm

from utils.customerDataset import CustomerDataset

dataset_root_path = CustomerConfig.TRAIN_DATASET
img_floder =os.path.join(dataset_root_path, "imgs")
mask_floder = os.path.join(dataset_root_path, "mask")
yaml_floder = os.path.join(dataset_root_path, "yaml")
imglist = os.listdir(img_floder)

config = CustomerConfig()

count = len(imglist)
dataset = CustomerDataset()
dataset.load_shapes(config.NAME, len(imglist), config.CLASSES, img_floder, mask_floder, imglist, yaml_floder, train_mode=False)
dataset.prepare()


# 生成imageids
# TODO: 多线程多进程优化
image_ids = [id for id in dataset.image_ids]
for imageid in tqdm(image_ids):
    dataset.load_mask(imageid, train_mode=False)




