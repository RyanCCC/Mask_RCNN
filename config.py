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
    NUM_CLASSES = 1 + 1
    RPN_ANCHOR_SCALES = (16, 32, 64, 128, 256)
    # IMAGE_MIN_DIM = 512
    # IMAGE_MAX_DIM = 512
    CLASSES = get_class(r'.\data\building.names')
    TRAIN_DATASET = './train_data'
    PRETRAIN_MODEL = "model/mask_rcnn_coco.h5"
    LEARNING_RATE = 1e-5
    EPOCH = 100

class InferenceConfig(Config):
    NAME = 'Customer'
    RPN_ANCHOR_SCALES = (16, 32, 64, 128, 256)
    IMAGE_MIN_DIM = 512
    IMAGE_MAX_DIM = 512
    model = r'.\logs\village_building.h5'
    classes_path = r'.\data\building.names'
