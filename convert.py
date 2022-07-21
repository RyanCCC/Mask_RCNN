'''
Convert weight model to pb or h5 or onnx
'''

import argparse
import tf2onnx
import tensorflow as tf
from mrcnn.mask_rcnn import MASK_RCNN


def parser_opt():
    parser = argparse.ArgumentParser(description="Convert Mask RCNN model")
    parser.add_argument('--weight', type=str, help='model weight', default='')
    parser.add_argument('--label', type=str,help='label file', default='')
    parser.add_argument('--saved_pb', action='store_true', help='save pb model to current directory')
    parser.add_argument('--saved_pb_dir', type=str, default='./save_model', help='save pb file if needed. Default:save_model')
    
    parser.add_argument('--saved_model', type=str, help='Tensorflow saved_model', default='')
    parser.add_argument('--save_onnx', type=str, help='save onnx model name', required=True, default='')
    parser.add_argument('--opset', type=int, default=12, help='ONNX: opset version')
    parser.add_argument('--flag', action='store_true', help='True:Tensoflow model, False:Tensorflow weights')
    
    return parser

def main(args):
    save_path = args.save_onnx
    opset = args.opset
    if args.flag:
        '''
        加载模型并导出onnx模型
        '''
        saved_model = args.saved_model
        assert len(saved_model) > 0, 'saved_model cannot be none or empty.'
        maskrcnn_model = tf.keras.models.load_model(saved_model)
        model_proto, _ = tf2onnx.convert.from_keras(maskrcnn_model, opset=opset, output_path=save_path)
        output_names = [n.name for n in model_proto.graph.output]
        print(output_names)
    else:
        print('Convert Tensorflow saved model to ONNX')
        weights = args.weight
        class_path = args.label
        assert len(weights) > 0, 'weights cannot be none or empty.'
        assert len(class_path) > 0, 'classes path doesn\'t exists.'
        mask_rcnn = MASK_RCNN(model=weights, classes_path=class_path, confidence=0.7)

        save_pb = args.saved_pb
        if save_pb:
            save_name = args.saved_pb_dir
            assert len(save_name) > 0, 'save_name cannot be none or empty.'
            mask_rcnn.model.save(save_name, save_format='tf')

        model_proto, _ = tf2onnx.convert.from_keras(mask_rcnn.model, opset=opset, output_path=save_path)
        output_names = [n.name for n in model_proto.graph.output]
        print(f'Model output names: ',output_names)

if __name__ == '__main__':
    parser = parser_opt()
    args = parser.parse_args()
    main(args=args)
