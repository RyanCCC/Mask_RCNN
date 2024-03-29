import os
from PIL import Image
import numpy as np
from .mrcnn import get_model
from utils.config import Config
from utils.anchors import get_anchors
from utils.utils import mold_inputs,unmold_detections
from utils import visualize
import tensorflow as tf
from config import InferenceConfig

# tf.compat.v1.disable_eager_execution()

class MASK_RCNN(object):

    def __init__(self, **kwargs):
        self.model_path = kwargs['model']
        self.classes_path= kwargs['classes_path']
        self.confidence = kwargs['confidence']

        # 使用coco数据集检测的时候，IMAGE_MIN_DIM=1024，IMAGE_MAX_DIM=1024, RPN_ANCHOR_SCALES=(32, 64, 128, 256, 512)
        self.RPN_ANCHOR_SCALES = InferenceConfig.RPN_ANCHOR_SCALES
        self.IMAGE_MIN_DIM = InferenceConfig.IMAGE_MIN_DIM
        self.IMAGE_MAX_DIM = InferenceConfig.IMAGE_MAX_DIM

        self.class_names = self.get_class()
        self.config = self._get_config()
        self.generate()

    def get_class(self):
        classes_path = os.path.expanduser(self.classes_path)
        with open(classes_path) as f:
            class_names = f.readlines()
        class_names = [c.strip() for c in class_names]
        class_names.insert(0,"BG")
        return class_names

    def _get_config(self):
        class InferenceConfig(Config):
            NUM_CLASSES = len(self.class_names)
            GPU_COUNT = 1
            IMAGES_PER_GPU = 1
            DETECTION_MIN_CONFIDENCE = self.confidence
            NAME = "Customer"
            RPN_ANCHOR_SCALES = self.RPN_ANCHOR_SCALES
            IMAGE_MIN_DIM = self.IMAGE_MIN_DIM
            IMAGE_MAX_DIM = self.IMAGE_MAX_DIM

        config = InferenceConfig()
        config.display()
        return config


    def generate(self):
        model_path = os.path.expanduser(self.model_path)
        assert model_path.endswith('.h5'), 'Keras model or weights must be a .h5 file.'
        
        # 计算总的种类
        self.num_classes = len(self.class_names)

        # 载入模型
        self.model = get_model(self.config, training=False)
        self.model.load_weights(self.model_path,by_name=True)
        # self.model.save('./maskrcnn_coco', save_format='tf')
    

    def detect_image(self, image, showBox = False, show_caption=False):
        image = [np.array(image)]
        molded_images, image_metas, windows = mold_inputs(self.config,image)

        image_shape = molded_images[0].shape
        anchors = get_anchors(self.config,image_shape)
        anchors = np.broadcast_to(anchors, (1,) + anchors.shape)

        detections, _, _, mrcnn_mask, _, _, _ =\
            self.model.predict([molded_images, image_metas, anchors], verbose=0)

        final_rois, final_class_ids, final_scores, final_masks =\
            unmold_detections(detections[0], mrcnn_mask[0],
                                    image[0].shape, molded_images[0].shape,
                                    windows[0])

        r = {
            "rois": final_rois,
            "class_ids": final_class_ids,
            "scores": final_scores,
            "masks": final_masks,
        }

        # 想要保存处理后的图片请查询plt保存图片的方法。
        drawed_image = visualize.display_instances(image[0], r['rois'], r['masks'], r['class_ids'], 
                                    self.class_names, r['scores'], show_bbox = showBox, captions=show_caption)
        # 处理mask 文件
        mask_image = np.any(r['masks'], axis=-1)
        mask_image = Image.fromarray(mask_image)
        return drawed_image, mask_image
    
    def get_detections(self, image):
        image = [np.array(image)]
        molded_images, image_metas, windows = mold_inputs(self.config,image)

        image_shape = molded_images[0].shape
        anchors = get_anchors(self.config,image_shape)
        anchors = np.broadcast_to(anchors, (1,) + anchors.shape)

        detections, _, _, mrcnn_mask, _, _, _ =\
            self.model.predict([molded_images, image_metas, anchors], verbose=0)

        final_rois, final_class_ids, final_scores, final_masks =\
            unmold_detections(detections[0], mrcnn_mask[0],
                                    image[0].shape, molded_images[0].shape,
                                    windows[0])

        r = {
            "rois": final_rois,
            "class_ids": final_class_ids,
            "scores": final_scores,
            "masks": final_masks,
        }
        return r
        
    def close_session(self):
        self.sess.close()