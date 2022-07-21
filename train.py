import os
from PIL import Image
import numpy as np
import random
import tensorflow as tf
from utils import visualize
from utils.anchors import get_anchors
from utils.utils import mold_inputs,unmold_detections
from mrcnn.mrcnn import get_model
from mrcnn.mrcnn_training import data_generator,load_image_gt
from utils.customerDataset import CustomerDataset
from config import CustomerConfig

# tf.compat.v1.disable_eager_execution()

def log(text, array=None):
    if array is not None:
        text = text.ljust(25)
        text += ("shape: {:20}  ".format(str(array.shape)))
        if array.size:
            text += ("min: {:10.5f}  max: {:10.5f}".format(array.min(),array.max()))
        else:
            text += ("min: {:10}  max: {:10}".format("",""))
        text += "  {}".format(array.dtype)
    print(text)



if __name__ == "__main__":
    learning_rate = CustomerConfig.LEARNING_RATE
    init_epoch = 0
    epoch = CustomerConfig.EPOCH
    
    dataset_root_path=CustomerConfig.TRAIN_DATASET
    img_floder =os.path.join(dataset_root_path, "imgs")
    mask_floder = os.path.join(dataset_root_path, "mask")
    yaml_floder = os.path.join(dataset_root_path, "yaml")
    imglist = os.listdir(img_floder)

    count = len(imglist)
    np.random.seed(10101)
    np.random.shuffle(imglist)
    train_imglist = imglist[:int(count*0.9)]
    val_imglist = imglist[int(count*0.9):]

    MODEL_DIR = "logs"

    COCO_MODEL_PATH = CustomerConfig.PRETRAIN_MODEL
    config = CustomerConfig()
    # 计算训练集和验证集长度
    config.STEPS_PER_EPOCH = len(train_imglist)//config.IMAGES_PER_GPU
    config.VALIDATION_STEPS = len(val_imglist)//config.IMAGES_PER_GPU
    config.display()

    # 训练数据集准备
    dataset_train = CustomerDataset()
    dataset_train.load_shapes(config.NAME,len(train_imglist), config.CLASSES, img_floder, mask_floder, train_imglist, yaml_floder, train_mode=True)
    dataset_train.prepare()

    # 验证数据集准备
    dataset_val = CustomerDataset()
    dataset_val.load_shapes(config.NAME,len(val_imglist), config.CLASSES, img_floder, mask_floder, val_imglist, yaml_floder, train_mode=True)
    dataset_val.prepare()

    # 获得训练模型
    model = get_model(config, training=True)
    model.summary()
    model.load_weights(COCO_MODEL_PATH,by_name=True,skip_mismatch=True)

    # 数据生成器
    train_generator = data_generator(dataset_train, config, shuffle=True,
                                        batch_size=config.BATCH_SIZE)
    val_generator = data_generator(dataset_val, config, shuffle=True,
                                    batch_size=config.BATCH_SIZE)

    # 设置callbacks
    tensorboard = tf.keras.callbacks.TensorBoard(log_dir=MODEL_DIR,histogram_freq=0, write_graph=True, write_images=False)
    model_ckp= tf.keras.callbacks.ModelCheckpoint(os.path.join(MODEL_DIR, "building_new.h5"),verbose=0, save_weights_only=True)
    early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0, patience=5, verbose=1)
    learning_rate_reduce = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.3, patience=3, verbose=1)
    callbacks = [tensorboard, model_ckp, early_stop, learning_rate_reduce]
    
    # callbacks = [
    #     tf.keras.callbacks.TensorBoard(log_dir=MODEL_DIR,
    #                                 histogram_freq=0, write_graph=True, write_images=False),
    #     tf.keras.callbacks.ModelCheckpoint(os.path.join(MODEL_DIR, "epoch{epoch:03d}_loss{loss:.3f}_val_loss{val_loss:.3f}.h5"),
    #                                     verbose=0, save_weights_only=True),
    # ]


    if True:
        log("\nStarting at epoch {}. LR={}\n".format(init_epoch, learning_rate))
        log("Checkpoint Path: {}".format(MODEL_DIR))

        # 使用的优化器是
        optimizer = tf.keras.optimizers.Adam(lr=learning_rate, clipnorm=config.GRADIENT_CLIP_NORM)

        # 设置一下loss信息
        loss_names = [
            "rpn_class_loss",  "rpn_bbox_loss",
            "mrcnn_class_loss", "mrcnn_bbox_loss", "mrcnn_mask_loss"]
        for name in loss_names:
            layer = model.get_layer(name)
            if layer.output in model.losses:
                continue
            loss = (
                tf.reduce_mean(input_tensor=layer.output, keepdims=True)
                * config.LOSS_WEIGHTS.get(name, 1.))
            model.add_loss(loss)
        
        # Add L2 Regularization
        # Skip gamma and beta weights of batch normalization layers.
        reg_losses = [
            tf.keras.regularizers.l2(config.WEIGHT_DECAY)(w) / tf.cast(tf.size(input=w), tf.float32)
            for w in model.trainable_weights
            if 'gamma' not in w.name and 'beta' not in w.name]
        model.add_loss(tf.add_n(reg_losses))


        # 进行编译
        model.compile(
            optimizer=optimizer,
            loss=[None] * len(model.outputs)
        )

        # 用于显示训练情况
        for name in loss_names:
            if name in model.metrics_names:
                print(name)
                continue
            layer = model.get_layer(name)
            model.metrics_names.append(name)
            loss = (
                tf.reduce_mean(input_tensor=layer.output, keepdims=True)
                * config.LOSS_WEIGHTS.get(name, 1.))
            model.add_metric(loss, name=name, aggregation='mean')


        model.fit_generator(
            train_generator,
            initial_epoch=init_epoch,
            epochs=epoch,
            steps_per_epoch=config.STEPS_PER_EPOCH,
            callbacks=callbacks,
            validation_data=val_generator,
            validation_steps=config.VALIDATION_STEPS,
            max_queue_size=100
        )
