from utils.config import Config
import os

def get_class(classes_path):
    classes_path = os.path.expanduser(classes_path)
    with open(classes_path) as f:
        class_names = f.readlines()
    class_names = [c.strip() for c in class_names]
    return class_names

class CustomerConfig(Config):
    NAME = "Customer"
    GPU_COUNT = 1
    # 应该通过设置IMAGES_PER_GPU来设置BATCH的大小，而不是下面的BATCH_SIZE
    # BATCHS_SIZE自动设置为IMAGES_PER_GPU*GPU_COUNT
    # 请各位注意哈！
    IMAGES_PER_GPU = 1
    NUM_CLASSES = 1 + 3
    RPN_ANCHOR_SCALES = (16, 32, 64, 128, 256)
    IMAGE_MIN_DIM = 512
    IMAGE_MAX_DIM = 512
    CLASSES = get_class('./data/shapes.names')
    TRAIN_DATASET = './train_dataset'
    PRETRAIN_MODEL = "model/mask_rcnn_coco.h5"
    LEARNING_RATE = 1e-5
    EPOCH = 100

class InferenceConfig(Config):
    NAME = 'Customer'
    RPN_ANCHOR_SCALES = (16, 32, 64, 128, 256)
    IMAGE_MIN_DIM = 512
    IMAGE_MAX_DIM = 512
    model = './logs/test.h5'
    classes_path = './data/shapes.names'
