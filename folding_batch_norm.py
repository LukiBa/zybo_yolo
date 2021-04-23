# -*- coding: utf-8 -*-
"""
Created on Wed Apr  7 08:54:27 2021

@author: lukas
"""

import argparse
import pathlib
import tensorflow as tf
from absl import app, logging
from core.yolov4 import YOLO, decode, filter_boxes
import core.utils as utils
from core.config import cfg
import core.backbone as backbone

def _create_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', type=str, default='./weights/yolov3-tiny.weights', help='path to weights file')
    parser.add_argument('--output',  type=str, default='./checkpoints/yolov3-tiny-416-fb', help='path to output')
    parser.add_argument('--output_backbone',  type=str, default='./checkpoints/yolov3-tiny-416-backbone', help='path to output')
    parser.add_argument('--tiny', type=bool, default=True, help='is yolo-tiny or not')
    parser.add_argument('--input_size', type=int, default=320, help='define input size of export model')
    parser.add_argument('--model', type=str, default='yolov3', help='yolov3 or yolov4')
    parser.add_argument('--score_thres', type=float, default=0.2, help='define score threshold')
    return parser.parse_args()

def save_tf(flags):
    STRIDES, ANCHORS, NUM_CLASS, XYSCALE = utils.load_config(flags.model,flags.tiny)
    
    input_layer = tf.keras.layers.Input([flags.input_size, flags.input_size, 3],name='input_0')
    if flags.model == 'yolov3':
        if flags.tiny:
            bckbn = backbone.darknet53_tiny_folding_bn
        else:
            raise NotImplementedError("Yolov3 folding batch norm not implemented yet. Simply copy backbone and set bn to False.")
    else:
        if flags.tiny:
            bckbn = backbone.cspdarknet53_tiny_folding_bn
        else:
            raise NotImplementedError("Yolov4 folding batch norm not implemented yet. Simply copy backbone and set bn to False.")        
    
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
    model.summary()
    utils.load_weights_folding_batchnorm(model, flags.weights, flags.model, flags.tiny)
    model.summary()
    outpath = pathlib.Path(flags.output).absolute()
    model.save(str(outpath))    
    
    outpath = pathlib.Path(flags.output_backbone).absolute()
    backbone_model = tf.keras.Model(input_layer,bckbn(input_layer))
    for layer in backbone_model.layers:
        if (isinstance(layer,tf.keras.layers.Conv2D)):
            layer.set_weights(model.get_layer(layer.name).get_weights())
    backbone_model.save(str(outpath)) 
    

def main(flags):
  save_tf(flags)

if __name__ == '__main__':
    flags = _create_parser()
    main(flags)
