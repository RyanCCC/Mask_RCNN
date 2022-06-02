from PIL import Image
import tensorflow as tf
import numpy as np
from utils.anchors import get_anchors
from utils.utils import mold_inputs,unmold_detections
from utils import visualize
from utils.config import Config
import os
from glob import glob
import json
import base64
import io
from tqdm import tqdm

'''
模型的自动标注代码
'''

img_pattern  = './todo/*.jpg'
model_path = './model/village_building'
class_path = './tmp.names'

class intEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.intc):
            return float(obj)
    
        return json.JSONEncoder.default(self, obj)

def get_class(classes_path):
        classes_path = os.path.expanduser(classes_path)
        with open(classes_path) as f:
            class_names = f.readlines()
        class_names = [c.strip() for c in class_names]
        class_names.insert(0,"BG")
        return class_names

def img2base64(image_path):
    with open(image_path, 'rb') as f:
        image = f.read()
    image_base64 = str(base64.b64encode(image), encoding='utf-8')
    return image_base64 

class_names = get_class(class_path)
def get_config():
    class InferenceConfig(Config):
        NUM_CLASSES = len(class_names)
        GPU_COUNT = 1
        IMAGES_PER_GPU = 1
        DETECTION_MIN_CONFIDENCE = 0.8
        NAME = "Customer"
        RPN_ANCHOR_SCALES =  (16, 32, 64, 128, 256)
        IMAGE_MIN_DIM = 512
        IMAGE_MAX_DIM = 512
        IMAGE_SHAPE =  [512, 512 ,3]

    config = InferenceConfig()
    return config

InferenceConfig = get_config()
model = tf.keras.models.load_model(model_path)

# 获取输入的anchor


for image_name in tqdm(glob(img_pattern)):
    image = Image.open(image_name).convert('RGB')
    imageWidth, imageHeight = image.size
    image = [np.array(image)]
    molded_images, image_metas, windows = mold_inputs(InferenceConfig,image)
    image_shape = molded_images[0].shape
    anchors = get_anchors(InferenceConfig,image_shape)
    anchors = np.broadcast_to(anchors, (1,) + anchors.shape)
    detections, _, _, mrcnn_mask, _, _, _ =model.predict([molded_images, image_metas, anchors], verbose=0)
    final_rois, final_class_ids, final_scores, final_masks =unmold_detections(detections[0], mrcnn_mask[0],image[0].shape, molded_images[0].shape,windows[0])

    r = {
        "rois": final_rois,
        "class_ids": final_class_ids,
        "scores": final_scores,
        "masks": final_masks,
    }


    drawed_image, shapes = visualize.display_instances(image[0], r['rois'], r['masks'], r['class_ids'], 
                                    class_names, r['scores'])

    # 将数据保存在json文件中
    json_info = {}
    json_info['version'] = "4.5.7"
    json_info['flags'] = {}
    json_info['shapes'] = []
    json_info['imagePath'] = os.path.basename(image_name)

    json_info['imageData'] = img2base64(image_name)
    json_info['imageHeight'] = imageHeight
    json_info['imageWidth'] = imageWidth
    # 处理返回的shapes
    for index, (label, points) in  enumerate(shapes):
        tmp = {}
        if len(points[0])<=2:
            continue
        tmp['label'] = label+f'.{index:03d}'
        tmp['points'] = []
        tmp['group_id'] = None
        tmp['shape_type'] = "polygon"
        tmp['flags'] = {}
        for point in points[0]:
            tmp['points'] .append(list(point[0]))
        json_info['shapes'].append(tmp)
    
    # 保存.json文件
    path = os.path.dirname(image_name)
    suffix = '.json'
    json_filename = os.path.join(path, os.path.splitext(json_info['imagePath'])[0]+suffix)
    with open(json_filename, 'w') as f:
        json.dump(json_info, f, ensure_ascii=False, indent=2, cls=intEncoder)

    # drawed_image.save('./tmp/6.jpg')
    # drawed_image.show()