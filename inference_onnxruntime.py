import os
import numpy as np
from utils.config import Config
from utils.anchors import get_anchors
from utils.utils import mold_inputs,unmold_detections
from utils.config import Config
import colorsys
import onnxruntime as ort
from PIL import Image


class InferenceConfig(Config):
    NAME = 'Customer'
    RPN_ANCHOR_SCALES = (16, 32, 64, 128, 256)
    IMAGE_MIN_DIM = 512
    IMAGE_MAX_DIM = 512
    model = './village_building_20221107.onnx'
    classes_path = './data/building.names'

def random_colors(N, bright=True):
    """
    生成随机颜色
    """
    brightness = 1.0 if bright else 0.7
    hsv = [(i / N, 1, brightness) for i in range(N)]
    colors = list(map(lambda c: colorsys.hsv_to_rgb(*c), hsv))
    return colors

def apply_mask(image, mask, color, alpha=0.5):
    """
    打上mask图标
    """
    for c in range(3):
        image[:, :, c] = np.where(mask == 1,
                                  image[:, :, c] *
                                  (1 - alpha) + alpha * color[c] * 255,
                                  image[:, :, c])
    return image

class MASK_RCNN(object):
    _defaults = {
        "model_path": InferenceConfig.model,
        "classes_path": InferenceConfig.classes_path,
        "confidence": 0.7,
        # 使用coco数据集检测的时候，IMAGE_MIN_DIM=1024，IMAGE_MAX_DIM=1024, RPN_ANCHOR_SCALES=(32, 64, 128, 256, 512)
        "RPN_ANCHOR_SCALES": InferenceConfig.RPN_ANCHOR_SCALES,
        "IMAGE_MIN_DIM": InferenceConfig.IMAGE_MIN_DIM,
        "IMAGE_MAX_DIM": InferenceConfig.IMAGE_MAX_DIM,
    }

    @classmethod
    def get_defaults(cls, n):
        if n in cls._defaults:
            return cls._defaults[n]
        else:
            return "Unrecognized attribute name '" + n + "'"

    def __init__(self, **kwargs):
        self.__dict__.update(self._defaults)
        self.class_names = self._get_class()
        self.config = self._get_config()
        self.generate()

    def _get_class(self):
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

        return config

    def generate(self):
        model_path = os.path.expanduser(self.model_path)
        
        # 计算总的种类
        self.num_classes = len(self.class_names)

        # 载入模型，如果原来的模型里已经包括了模型结构则直接载入。
        self.model = ort.InferenceSession(model_path)
        self.outputs_names = ['mrcnn_detection', 'mrcnn_class', 'mrcnn_bbox', 'mrcnn_mask', 'ROI', 'rpn_class', 'rpn_bbox']
    
    def detect_image(self, image):
        image = [np.array(image)]
        molded_images, image_metas, windows = mold_inputs(self.config,image)

        image_shape = molded_images[0].shape
        anchors = get_anchors(self.config,image_shape)
        anchors = np.broadcast_to(anchors, (1,) + anchors.shape)

        detections, _, _, mrcnn_mask, _, _, _ =\
            self.model.run(self.outputs_names, {"input_image":molded_images.astype(np.float32), "input_image_meta":image_metas.astype(np.float32), "input_anchors":anchors.astype(np.float32)})

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
        # 生成mask图像
        mask_image = np.zeros_like(image[0], np.uint8)
        masks = r['masks']
        N = r['rois'].shape[0]
        for i in range(N):
            mask = masks[:, :, i]
            color = (1.0, 0.0, 0.0)
            mask_image = apply_mask(mask_image, mask, color, alpha=1)
            padded_mask = np.zeros(
            (mask.shape[0] + 2, mask.shape[1] + 2), dtype=np.uint8)
            padded_mask[1:-1, 1:-1] = mask
        return mask_image
    
if __name__ == '__main__':
    mask_rcnn = MASK_RCNN()
    img = './samples/2021-11-01_112614.jpg'
    image = Image.open(img)
    r_image = mask_rcnn.detect_image(image)
    img = Image.fromarray(r_image)
    img = Image.blend(img, image, 0.7)
    # img.save('./test.png')
    img.show()
    