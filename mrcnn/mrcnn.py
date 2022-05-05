from .layers import ProposalLayer,PyramidROIAlign,DetectionLayer,DetectionTargetLayer
from .mrcnn_training import *
from utils.anchors import get_anchors
from utils.utils import norm_boxes_graph,parse_image_meta_graph
import numpy as np
import tensorflow as tf
import tensorflow.keras as keras
import tensorflow.keras.backend as K
import tensorflow.keras.layers as KL
import tensorflow.keras.utils as KU
from tensorflow.python.eager import context
import tensorflow.keras.models as KM
from mrcnn.restnet import get_resnet


tf.compat.v1.disable_eager_execution()


def rpn_graph(feature_map, anchors_per_location, anchor_stride):
    
    shared = KL.Conv2D(512, (3, 3), padding='same', activation='relu',strides=anchor_stride,
                       name='rpn_conv_shared')(feature_map)
    
    x = KL.Conv2D(2 * anchors_per_location, (1, 1), padding='valid',
                  activation='linear', name='rpn_class_raw')(shared)
    # batch_size,num_anchors,2
    # 代表这个先验框对应的类
    rpn_class_logits = KL.Reshape([-1,2])(x)

    rpn_probs = KL.Activation(
        "softmax", name="rpn_class_xxx")(rpn_class_logits)
    
    x = KL.Conv2D(anchors_per_location * 4, (1, 1), padding="valid",
                  activation='linear', name='rpn_bbox_pred')(shared)
    # batch_size,num_anchors,4
    # 这个先验框的调整参数
    rpn_bbox = KL.Reshape([-1,4])(x)

    return [rpn_class_logits, rpn_probs, rpn_bbox]


def build_rpn_model(anchor_stride, anchors_per_location, depth):
    """Builds a Keras model of the Region Proposal Network.
    It wraps the RPN graph so it can be used multiple times with shared
    weights.

    anchors_per_location: number of anchors per pixel in the feature map
    anchor_stride: Controls the density of anchors. Typically 1 (anchors for
                   every pixel in the feature map), or 2 (every other pixel).
    depth: Depth of the backbone feature map.

    Returns a Keras Model object. The model outputs, when called, are:
    rpn_class_logits: [batch, H * W * anchors_per_location, 2] Anchor classifier logits (before softmax)
    rpn_probs: [batch, H * W * anchors_per_location, 2] Anchor classifier probabilities.
    rpn_bbox: [batch, H * W * anchors_per_location, (dy, dx, log(dh), log(dw))] Deltas to be
                applied to anchors.
    """
    input_feature_map = KL.Input(shape=[None, None, depth],
                                 name="input_rpn_feature_map")
    outputs = rpn_graph(input_feature_map, anchors_per_location, anchor_stride)
    return KM.Model([input_feature_map], outputs, name="rpn_model")



def fpn_classifier_graph(rois, feature_maps, image_meta,
                         pool_size, num_classes, train_bn=True,
                         fc_layers_size=1024):
    # ROI Pooling，利用建议框在特征层上进行截取
    # Shape: [batch, num_rois, POOL_SIZE, POOL_SIZE, channels]
    x = PyramidROIAlign([pool_size, pool_size],
                        name="roi_align_classifier")([rois, image_meta] + feature_maps)

    # Shape: [batch, num_rois, 1, 1, fc_layers_size]，相当于两次全连接
    x = KL.TimeDistributed(KL.Conv2D(fc_layers_size, (pool_size, pool_size), padding="valid"),
                           name="mrcnn_class_conv1")(x)
    x = KL.TimeDistributed(KL.BatchNormalization(), name='mrcnn_class_bn1')(x, training=train_bn)
    x = KL.Activation('relu')(x)

    # Shape: [batch, num_rois, 1, 1, fc_layers_size]
    x = KL.TimeDistributed(KL.Conv2D(fc_layers_size, (1, 1)),
                           name="mrcnn_class_conv2")(x)
    x = KL.TimeDistributed(KL.BatchNormalization(), name='mrcnn_class_bn2')(x, training=train_bn)
    x = KL.Activation('relu')(x)

    # Shape: [batch, num_rois, fc_layers_size]
    shared = KL.Lambda(lambda x: K.squeeze(K.squeeze(x, 3), 2),
                       name="pool_squeeze")(x)

    # Classifier head
    # 这个的预测结果代表这个先验框内部的物体的种类
    mrcnn_class_logits = KL.TimeDistributed(KL.Dense(num_classes),
                                            name='mrcnn_class_logits')(shared)
    mrcnn_probs = KL.TimeDistributed(KL.Activation("softmax"),
                                     name="mrcnn_class")(mrcnn_class_logits)


    # BBox head
    # 这个的预测结果会对先验框进行调整
    # [batch, num_rois, NUM_CLASSES * (dy, dx, log(dh), log(dw))]
    x = KL.TimeDistributed(KL.Dense(num_classes * 4, activation='linear'),
                           name='mrcnn_bbox_fc')(shared)
    # Reshape to [batch, num_rois, NUM_CLASSES, (dy, dx, log(dh), log(dw))]
    s = K.int_shape(x)
    if s[1] is None:
        mrcnn_bbox = KL.Reshape((-1, num_classes, 4), name="mrcnn_bbox")(x)
    else:
        mrcnn_bbox = KL.Reshape((s[1], num_classes, 4), name="mrcnn_bbox")(x)

    return mrcnn_class_logits, mrcnn_probs, mrcnn_bbox



def build_fpn_mask_graph(rois, feature_maps, image_meta,
                         pool_size, num_classes, train_bn=True):
    # ROI Pooling，利用建议框在特征层上进行截取
    # Shape: [batch, num_rois, MASK_POOL_SIZE, MASK_POOL_SIZE, channels]
    x = PyramidROIAlign([pool_size, pool_size],
                        name="roi_align_mask")([rois, image_meta] + feature_maps)

    # Shape: [batch, num_rois, MASK_POOL_SIZE, MASK_POOL_SIZE, channels]
    x = KL.TimeDistributed(KL.Conv2D(256, (3, 3), padding="same"),
                           name="mrcnn_mask_conv1")(x)
    x = KL.TimeDistributed(KL.BatchNormalization(),
                           name='mrcnn_mask_bn1')(x, training=train_bn)
    x = KL.Activation('relu')(x)

    # Shape: [batch, num_rois, MASK_POOL_SIZE, MASK_POOL_SIZE, channels]
    x = KL.TimeDistributed(KL.Conv2D(256, (3, 3), padding="same"),
                           name="mrcnn_mask_conv2")(x)
    x = KL.TimeDistributed(KL.BatchNormalization(),
                           name='mrcnn_mask_bn2')(x, training=train_bn)
    x = KL.Activation('relu')(x)

    # Shape: [batch, num_rois, MASK_POOL_SIZE, MASK_POOL_SIZE, channels]
    x = KL.TimeDistributed(KL.Conv2D(256, (3, 3), padding="same"),
                           name="mrcnn_mask_conv3")(x)
    x = KL.TimeDistributed(KL.BatchNormalization(),
                           name='mrcnn_mask_bn3')(x, training=train_bn)
    x = KL.Activation('relu')(x)

    # Shape: [batch, num_rois, MASK_POOL_SIZE, MASK_POOL_SIZE, channels]
    x = KL.TimeDistributed(KL.Conv2D(256, (3, 3), padding="same"),
                           name="mrcnn_mask_conv4")(x)
    x = KL.TimeDistributed(KL.BatchNormalization(),
                           name='mrcnn_mask_bn4')(x, training=train_bn)
    x = KL.Activation('relu')(x)

    # Shape: [batch, num_rois, 2xMASK_POOL_SIZE, 2xMASK_POOL_SIZE, channels]
    x = KL.TimeDistributed(KL.Conv2DTranspose(256, (2, 2), strides=2, activation="relu"),
                           name="mrcnn_mask_deconv")(x)
    # 反卷积后再次进行一个1x1卷积调整通道，使其最终数量为numclasses，代表分的类
    x = KL.TimeDistributed(KL.Conv2D(num_classes, (1, 1), strides=1, activation="sigmoid"),
                           name="mrcnn_mask")(x)
    return x


def get_model(config, training):
    # Image size must be dividable by 2 multiple times
    h, w = config.IMAGE_SHAPE[:2]
    if h / 2**6 != int(h / 2**6) or w / 2**6 != int(w / 2**6):
        raise Exception("Image size must be dividable by 2 at least 6 times "
                            "to avoid fractions when downscaling and upscaling."
                            "For example, use 256, 320, 384, 448, 512, ... etc. ")

    # Inputs
    input_image = KL.Input(
            shape=[None, None, config.IMAGE_SHAPE[2]], name="input_image")
    input_image_meta = KL.Input(shape=[config.IMAGE_META_SIZE],
                                    name="input_image_meta")

    if training:
        input_rpn_match = KL.Input(
                shape=[None, 1], name="input_rpn_match", dtype=tf.int32)
        input_rpn_bbox = KL.Input(
                shape=[None, 4], name="input_rpn_bbox", dtype=tf.float32)

        # Detection GT (class IDs, bounding boxes, and masks)
        # 1. GT Class IDs (zero padded)
        input_gt_class_ids = KL.Input(
                shape=[None], name="input_gt_class_ids", dtype=tf.int32)
        # 2. GT Boxes in pixels (zero padded)
        # [batch, MAX_GT_INSTANCES, (y1, x1, y2, x2)] in image coordinates
        input_gt_boxes = KL.Input(
                shape=[None, 4], name="input_gt_boxes", dtype=tf.float32)
        # Normalize coordinates
        gt_boxes = KL.Lambda(lambda x: norm_boxes_graph(
                x, K.shape(input_image)[1:3]))(input_gt_boxes)

        # mask语义分析信息
        # [batch, height, width, MAX_GT_INSTANCES]
        if config.USE_MINI_MASK:
            input_gt_masks = KL.Input(shape=[config.MINI_MASK_SHAPE[0],config.MINI_MASK_SHAPE[1], None],name="input_gt_masks", dtype=bool)
        else:
            input_gt_masks = KL.Input(shape=[config.IMAGE_SHAPE[0], config.IMAGE_SHAPE[1], None],name="input_gt_masks", dtype=bool)
        # 设置anchor
        anchors = get_anchors(config,config.IMAGE_SHAPE)
        # 拓展anchors的shape，第一个维度拓展为batch_size
        anchors = np.broadcast_to(anchors, (config.BATCH_SIZE,) + anchors.shape)
        # 将anchors转化成tensor的形式
        class ConstLayer(tf.keras.layers.Layer):
            def __init__(self, x, name=None):
                super(ConstLayer, self).__init__(name=name)
                self.x = tf.Variable(x)

            def call(self, input):
                return self.x

        anchors = ConstLayer(anchors, name="anchors")(input_image)

    else:
        input_anchors = KL.Input(shape=[None, 4], name="input_anchors")
        anchors = input_anchors

    # 获得Resnet里的压缩程度不同的一些层
    _, C2, C3, C4, C5 = get_resnet(input_image, stage5=True, train_bn=config.TRAIN_BN)

    # 组合成特征金字塔的结构
    # P5长宽共压缩了5次
    # Height/32,Width/32,256
    P5 = KL.Conv2D(config.TOP_DOWN_PYRAMID_SIZE, (1, 1), name='fpn_c5p5')(C5)
    # P4长宽共压缩了4次
    # Height/16,Width/16,256
    P4 = KL.Add(name="fpn_p4add")([
        KL.UpSampling2D(size=(2, 2), name="fpn_p5upsampled")(P5),
        KL.Conv2D(config.TOP_DOWN_PYRAMID_SIZE, (1, 1), name='fpn_c4p4')(C4)])
    # P4长宽共压缩了3次
    # Height/8,Width/8,256
    P3 = KL.Add(name="fpn_p3add")([
        KL.UpSampling2D(size=(2, 2), name="fpn_p4upsampled")(P4),
        KL.Conv2D(config.TOP_DOWN_PYRAMID_SIZE, (1, 1), name='fpn_c3p3')(C3)])
    # P4长宽共压缩了2次
    # Height/4,Width/4,256
    P2 = KL.Add(name="fpn_p2add")([
        KL.UpSampling2D(size=(2, 2), name="fpn_p3upsampled")(P3),
        KL.Conv2D(config.TOP_DOWN_PYRAMID_SIZE, (1, 1), name='fpn_c2p2')(C2)])
        
    # 各自进行一次256通道的卷积，此时P2、P3、P4、P5通道数相同
    # Height/4,Width/4,256
    P2 = KL.Conv2D(config.TOP_DOWN_PYRAMID_SIZE, (3, 3), padding="SAME", name="fpn_p2")(P2)
    # Height/8,Width/8,256
    P3 = KL.Conv2D(config.TOP_DOWN_PYRAMID_SIZE, (3, 3), padding="SAME", name="fpn_p3")(P3)
    # Height/16,Width/16,256
    P4 = KL.Conv2D(config.TOP_DOWN_PYRAMID_SIZE, (3, 3), padding="SAME", name="fpn_p4")(P4)
    # Height/32,Width/32,256
    P5 = KL.Conv2D(config.TOP_DOWN_PYRAMID_SIZE, (3, 3), padding="SAME", name="fpn_p5")(P5)
    # 在建议框网络里面还有一个P6用于获取建议框
    # Height/64,Width/64,256
    P6 = KL.MaxPooling2D(pool_size=(1, 1), strides=2, name="fpn_p6")(P5)

    # P2, P3, P4, P5, P6可以用于获取建议框
    rpn_feature_maps = [P2, P3, P4, P5, P6]
    # P2, P3, P4, P5用于获取mask信息
    mrcnn_feature_maps = [P2, P3, P4, P5]

    
    
    # anchors = KL.Lambda(lambda x: tf.Variable(anchors), name="anchors")(input_image)
    # 建立RPN模型
    rpn = build_rpn_model(config.RPN_ANCHOR_STRIDE, len(config.RPN_ANCHOR_RATIOS), config.TOP_DOWN_PYRAMID_SIZE)

    if training:
        # Loop through pyramid layers
        layer_outputs = []  # list of lists
        for p in rpn_feature_maps:
            layer_outputs.append(rpn([p]))

        # 获得RPN网络的预测结果，进行格式调整，把五个特征层的结果进行堆叠
        output_names = ["rpn_class_logits", "rpn_class", "rpn_bbox"]
        outputs = list(zip(*layer_outputs))
        outputs = [KL.Concatenate(axis=1, name=n)(list(o))
                       for o, n in zip(outputs, output_names)]

        rpn_class_logits, rpn_class, rpn_bbox = outputs
    else:
        rpn_class_logits, rpn_class, rpn_bbox = [],[],[]

        # 获得RPN网络的预测结果，进行格式调整，把五个特征层的结果进行堆叠
        for p in rpn_feature_maps:
            logits,classes,bbox = rpn([p])
            rpn_class_logits.append(logits)
            rpn_class.append(classes)
            rpn_bbox.append(bbox)

        rpn_class_logits = KL.Concatenate(axis=1,name="rpn_class_logits")(rpn_class_logits)
        rpn_class =KL.Concatenate(axis=1,name="rpn_class")(rpn_class)
        rpn_bbox = KL.Concatenate(axis=1,name="rpn_bbox")(rpn_bbox)

    # 此时获得的rpn_class_logits、rpn_class、rpn_bbox的维度是
    # rpn_class_logits : Batch_size, num_anchors, 2
    # rpn_class : Batch_size, num_anchors, 2
    # rpn_bbox : Batch_size, num_anchors, 4
    proposal_count = config.POST_NMS_ROIS_TRAINING

    # Batch_size, proposal_count, 4
    rpn_rois = ProposalLayer(
            proposal_count=proposal_count,
            nms_threshold=config.RPN_NMS_THRESHOLD,
            name="ROI",
            config=config)([rpn_class, rpn_bbox, anchors])

    if not training:
        mrcnn_class_logits, mrcnn_class, mrcnn_bbox =\
        fpn_classifier_graph(rpn_rois, mrcnn_feature_maps, input_image_meta,
                                config.POOL_SIZE, config.NUM_CLASSES,
                                train_bn=config.TRAIN_BN,
                                fc_layers_size=config.FPN_CLASSIF_FC_LAYERS_SIZE)
    
        detections = DetectionLayer(config, name="mrcnn_detection")(
                    [rpn_rois, mrcnn_class, mrcnn_bbox, input_image_meta])
                
                
        detection_boxes = KL.Lambda(lambda x: x[..., :4])(detections)
        # 获得mask的结果
        mrcnn_mask = build_fpn_mask_graph(detection_boxes, mrcnn_feature_maps,
                                    input_image_meta,
                                    config.MASK_POOL_SIZE,
                                    config.NUM_CLASSES,
                                    train_bn=config.TRAIN_BN)

        # 作为输出
        model = KM.Model([input_image, input_image_meta, input_anchors],
                        [detections, mrcnn_class, mrcnn_bbox,
                            mrcnn_mask, rpn_rois, rpn_class, rpn_bbox],
                        name='mask_rcnn')
        return model

    active_class_ids = KL.Lambda(
        lambda x: parse_image_meta_graph(x)["active_class_ids"]
        )(input_image_meta)

    if not config.USE_RPN_ROIS:
        # 使用外部输入的建议框
        input_rois = KL.Input(shape=[config.POST_NMS_ROIS_TRAINING, 4],
                                name="input_roi", dtype=np.int32)
        # Normalize coordinates
        target_rois = KL.Lambda(lambda x: norm_boxes_graph(
            x, K.shape(input_image)[1:3]))(input_rois)
    else:
        # 利用预测到的建议框进行下一步的操作
        target_rois = rpn_rois

    """找到建议框的ground_truth
    Inputs:
    proposals: [batch, N, (y1, x1, y2, x2)]建议框
    gt_class_ids: [batch, MAX_GT_INSTANCES]每个真实框对应的类
    gt_boxes: [batch, MAX_GT_INSTANCES, (y1, x1, y2, x2)]真实框的位置
    gt_masks: [batch, height, width, MAX_GT_INSTANCES]真实框的语义分割情况
    Returns: 
    rois: [batch, TRAIN_ROIS_PER_IMAGE, (y1, x1, y2, x2)]内部真实存在目标的建议框
    target_class_ids: [batch, TRAIN_ROIS_PER_IMAGE]每个建议框对应的类
    target_deltas: [batch, TRAIN_ROIS_PER_IMAGE, (dy, dx, log(dh), log(dw)]每个建议框应该有的调整参数
    target_mask: [batch, TRAIN_ROIS_PER_IMAGE, height, width]每个建议框语义分割情况
    """
    rois, target_class_ids, target_bbox, target_mask =\
        DetectionTargetLayer(config, name="proposal_targets")([
            target_rois, input_gt_class_ids, gt_boxes, input_gt_masks])

    # 找到合适的建议框的classifier预测结果
    mrcnn_class_logits, mrcnn_class, mrcnn_bbox =\
        fpn_classifier_graph(rois, mrcnn_feature_maps, input_image_meta,
                                config.POOL_SIZE, config.NUM_CLASSES,
                                train_bn=config.TRAIN_BN,
                                fc_layers_size=config.FPN_CLASSIF_FC_LAYERS_SIZE)
    # 找到合适的建议框的mask预测结果
    mrcnn_mask = build_fpn_mask_graph(rois, mrcnn_feature_maps,
                                        input_image_meta,
                                        config.MASK_POOL_SIZE,
                                        config.NUM_CLASSES,
                                        train_bn=config.TRAIN_BN)

    output_rois = KL.Lambda(lambda x: x * 1, name="output_rois")(rois)

    # Losses
    rpn_class_loss = KL.Lambda(lambda x: rpn_class_loss_graph(*x), name="rpn_class_loss")(
        [input_rpn_match, rpn_class_logits])
    rpn_bbox_loss = KL.Lambda(lambda x: rpn_bbox_loss_graph(config, *x), name="rpn_bbox_loss")(
        [input_rpn_bbox, input_rpn_match, rpn_bbox])
    class_loss = KL.Lambda(lambda x: mrcnn_class_loss_graph(*x), name="mrcnn_class_loss")(
        [target_class_ids, mrcnn_class_logits, active_class_ids])
    bbox_loss = KL.Lambda(lambda x: mrcnn_bbox_loss_graph(*x), name="mrcnn_bbox_loss")(
        [target_bbox, target_class_ids, mrcnn_bbox])
    mask_loss = KL.Lambda(lambda x: mrcnn_mask_loss_graph(*x), name="mrcnn_mask_loss")(
        [target_mask, target_class_ids, mrcnn_mask])

    # Model
    inputs = [input_image, input_image_meta,
                input_rpn_match, input_rpn_bbox, input_gt_class_ids, input_gt_boxes, input_gt_masks]
                
    if not config.USE_RPN_ROIS:
        inputs.append(input_rois)
    outputs = [rpn_class_logits, rpn_class, rpn_bbox,
                mrcnn_class_logits, mrcnn_class, mrcnn_bbox, mrcnn_mask,
                rpn_rois, output_rois,
                rpn_class_loss, rpn_bbox_loss, class_loss, bbox_loss, mask_loss]
    model = KM.Model(inputs, outputs, name='mask_rcnn')
    return model