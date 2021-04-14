# -*- coding: utf-8 -*-
"""
Created on Wed Apr  7 08:54:27 2021

@author: lukas
"""

import argparse
import tensorflow as tf
from absl import app, logging
from core.yolov4 import YOLO, decode, filter_boxes
import core.utils as utils
from core.config import cfg

def _create_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', type=str, default='./weights/yolov4-tiny-best.weights', help='path to weights file')
    parser.add_argument('--output',  type=str, default='./checkpoints/yolov4-tiny-416-fb', help='path to output')
    parser.add_argument('--tiny', type=bool, default=True, help='is yolo-tiny or not')
    parser.add_argument('--input_size', type=int, default=416, help='define input size of export model')
    parser.add_argument('--model', type=str, default='yolov4', help='yolov3 or yolov4')
    parser.add_argument('--score_thres', type=float, default=0.2, help='define score threshold')
    return parser.parse_args()

def save_tf(flags):
    STRIDES, ANCHORS, NUM_CLASS, XYSCALE = utils.load_config(flags.model,flags.tiny)
    
    input_layer = tf.keras.layers.Input([flags.input_size, flags.input_size, 3])
    feature_maps_fbn = YOLO(input_layer, NUM_CLASS, flags.model, flags.tiny, fbn=True)
    bbox_tensors_fbn = []
    prob_tensors_fbn = []   
    
    if flags.tiny:
        for i, fm in enumerate(feature_maps_fbn):
            if i == 0:
                output_tensors_fbn = decode(fm, flags.input_size // 16, NUM_CLASS, STRIDES, ANCHORS, i, XYSCALE, 'tflite')
            else:
                output_tensors_fbn = decode(fm, flags.input_size // 32, NUM_CLASS, STRIDES, ANCHORS, i, XYSCALE, 'tflite')
            bbox_tensors_fbn.append(output_tensors_fbn[0])
            prob_tensors_fbn.append(output_tensors_fbn[1])          
          
    else:
        for i, fm in enumerate(feature_maps_fbn):
            if i == 0:
                output_tensors_fbn = decode(fm, flags.input_size // 8, NUM_CLASS, STRIDES, ANCHORS, i, XYSCALE, 'tflite')
            elif i == 1:
                output_tensors_fbn = decode(fm, flags.input_size // 16, NUM_CLASS, STRIDES, ANCHORS, i, XYSCALE, 'tflite')            
            else:
                output_tensors_fbn = decode(fm, flags.input_size // 32, NUM_CLASS, STRIDES, ANCHORS, i, XYSCALE, 'tflite')
            bbox_tensors_fbn.append(output_tensors_fbn[0])
            prob_tensors_fbn.append(output_tensors_fbn[1])             
          
           
    pred_bbox = tf.concat(bbox_tensors_fbn, axis=1)
    pred_prob = tf.concat(prob_tensors_fbn, axis=1)
    pred = (pred_bbox, pred_prob)
    model = tf.keras.Model(input_layer, pred) 
    utils.load_weights_folding_batchnorm(model, flags.weights, flags.model, flags.tiny)
    model.summary()
    model.save(flags.output)    

def main(flags):
  save_tf(flags)

if __name__ == '__main__':
    flags = _create_parser()
    main(flags)
