from mrcnn.mask_rcnn import MASK_RCNN
from PIL import Image

import tensorflow as tf
import numpy as np
from utils.anchors import get_anchors
from utils.utils import mold_inputs,unmold_detections
from utils import visualize
import os
from config import InferenceConfig

img = './images/20210604105809.jpg'

# def get_class(classes_path):
#         classes_path = os.path.expanduser(classes_path)
#         with open(classes_path) as f:
#             class_names = f.readlines()
#         class_names = [c.strip() for c in class_names]
#         class_names.insert(0,"BG")
#         return class_names


# model_path = './model/building'
# class_path = './data/building.names'
# class_names = get_class(class_path)

# def get_config():
#     class InferenceConfig(Config):
#         NUM_CLASSES = len(class_names)
#         GPU_COUNT = 1
#         IMAGES_PER_GPU = 1
#         DETECTION_MIN_CONFIDENCE = 0.7
#         NAME = "Customer"
#         RPN_ANCHOR_SCALES =  (16, 32, 64, 128, 256)
#         IMAGE_MIN_DIM = 512
#         IMAGE_MAX_DIM = 512
#         IMAGE_SHAPE =  [512, 512 ,3]

#     config = InferenceConfig()
#     config.display()
#     return config

# InferenceConfig = get_config()
# model = tf.keras.models.load_model(model_path)

# image = Image.open(img)
# image = [np.array(image)]

# molded_images, image_metas, windows = mold_inputs(InferenceConfig,image)

# image_shape = molded_images[0].shape
# anchors = get_anchors(InferenceConfig,image_shape)
# anchors = np.broadcast_to(anchors, (1,) + anchors.shape)
# detections, _, _, mrcnn_mask, _, _, _ =model.predict([molded_images, image_metas, anchors], verbose=0)
# final_rois, final_class_ids, final_scores, final_masks =unmold_detections(detections[0], mrcnn_mask[0],image[0].shape, molded_images[0].shape,windows[0])

# r = {
#     "rois": final_rois,
#     "class_ids": final_class_ids,
#     "scores": final_scores,
#     "masks": final_masks,
# }


# drawed_image = visualize.display_instances(image[0], r['rois'], r['masks'], r['class_ids'], 
#                                     class_names, r['scores'])
# drawed_image.save('6.jpg')
# drawed_image.show()

img = './images/2.jpg'
mask_rcnn = MASK_RCNN(model=InferenceConfig.model, classes_path=InferenceConfig.class_path, confidence=0.7)
img = Image.open(img)
drawed_image,mask_image = mask_rcnn.detect_image(image = img)
drawed_image.show()