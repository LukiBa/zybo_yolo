import tensorflow as tf
import argparse
from core.yolov4 import YOLO, decode, filter_boxes
import core.utils as utils
from core.config import cfg

def _create_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', type=str, default='./weights/yolov4-tiny-best.weights', help='path to weights file')
    parser.add_argument('--output', type=str, default='./checkpoints/yolov4-tiny-416', help='path to output')
    parser.add_argument('--tiny', type=bool, default=True, help='is yolo-tiny or not')
    parser.add_argument('--input_size', type=int, default=416, help='define input size of export model')
    parser.add_argument('--score_thres', type=float, default=0.2, help='score threshold')
    parser.add_argument('--framework', type=str, default='tflite', help='define what framework do you want to convert (tf, trt, tflite)')
    parser.add_argument('--model', type=str, default='yolov4', help='yolov3 or yolov4')
    return parser.parse_args()   

def save_tf(flags):
  STRIDES, ANCHORS, NUM_CLASS, XYSCALE = utils.load_config(flags.model,flags.tiny)

  input_layer = tf.keras.layers.Input([flags.input_size, flags.input_size, 3])
  feature_maps = YOLO(input_layer, NUM_CLASS, flags.model, flags.tiny)
  bbox_tensors = []
  prob_tensors = []
  if flags.tiny:
    for i, fm in enumerate(feature_maps):
      if i == 0:
        output_tensors = decode(fm, flags.input_size // 16, NUM_CLASS, STRIDES, ANCHORS, i, XYSCALE, flags.framework)
      else:
        output_tensors = decode(fm, flags.input_size // 32, NUM_CLASS, STRIDES, ANCHORS, i, XYSCALE, flags.framework)
      bbox_tensors.append(output_tensors[0])
      prob_tensors.append(output_tensors[1])
  else:
    for i, fm in enumerate(feature_maps):
      if i == 0:
        output_tensors = decode(fm, flags.input_size // 8, NUM_CLASS, STRIDES, ANCHORS, i, XYSCALE, flags.framework)
      elif i == 1:
        output_tensors = decode(fm, flags.input_size // 16, NUM_CLASS, STRIDES, ANCHORS, i, XYSCALE, flags.framework)
      else:
        output_tensors = decode(fm, flags.input_size // 32, NUM_CLASS, STRIDES, ANCHORS, i, XYSCALE, flags.framework)
      bbox_tensors.append(output_tensors[0])
      prob_tensors.append(output_tensors[1])
  pred_bbox = tf.concat(bbox_tensors, axis=1)
  pred_prob = tf.concat(prob_tensors, axis=1)
  if flags.framework == 'tflite':
    pred = (pred_bbox, pred_prob)
  else:
    boxes, pred_conf = filter_boxes(pred_bbox, pred_prob, score_threshold=flags.score_thres, input_shape=tf.constant([flags.input_size, flags.input_size]))
    pred = tf.concat([boxes, pred_conf], axis=-1)
  model = tf.keras.Model(input_layer, pred)
  utils.load_weights(model, flags.weights, flags.model, flags.tiny)
  model.summary()
  model.save(flags.output)

def main(flags):
  save_tf(flags)

if __name__ == '__main__':
    flags = _create_parser()
    main(flags)
