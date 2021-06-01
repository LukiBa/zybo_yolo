# -*- coding: utf-8 -*-
"""
Created on Sun Feb 21 11:40:32 2021

@author: lukas
"""

#%% import public modules
import argparse
import numpy as np
import matplotlib.pyplot as plt 
import pathlib
import timeit
from tensorflow.keras import optimizers
from tensorflow.keras import layers as keras_layer 
from tensorflow.keras import models as keras_models
import tensorflow.keras.backend as keras_backend
from tensorflow.keras.utils import plot_model
import tensorflow as tf

from core.yolov4 import YOLO, decode, filter_boxes
import core.utils as utils
from core.config import cfg
import core.backbone as backbone


#%% import custom modules 
import Intuitus
from Intuitus.models import Intuitus_Model
from Intuitus.layers import Conv2D_fl8
from Intuitus.core import float8, float6, MACC, float12
import IntuitusExtension as C_impl
#%% Parameter 
parser = argparse.ArgumentParser()
parser.add_argument('--weights', type=str, default='../PytorchYolo/parameters/int8_6', help='path to folder containing parameter')
parser.add_argument('--tiny', type=bool, default=True, help='is yolo-tiny or not')
parser.add_argument('--input_size', type=int, default=416, help='define input size of export model')
parser.add_argument('--model', type=str, default='yolov3', help='yolov3 or yolov4')
parser.add_argument('--score_thres', type=float, default=0.2, help='define score threshold')
parser.add_argument('--keras2torch_json',  type=str, default='./keras_2_torch_names.json', help='path name transaltion tabular json file')
parser.add_argument('--out_path',  type=str, default='./intuitus_commands/yolov3-tiny-commands', help='command output path')
flags=parser.parse_args()

hw_path = pathlib.Path(Intuitus.__file__).absolute().parents[3]  / "Hardware" 
out_path = pathlib.Path(flags.out_path).absolute()
# %% load pretrained model. Use example_nn.py for training the model
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
modelN = tf.keras.Model(input_layer, pred) 
modelN.summary()
utils.load_weights_torch_npy_fb(modelN, flags.weights, flags.model, flags.tiny, flags.keras2torch_json)

# %% Initialize Intuitus model using pretrained keras model
model = Intuitus_Model(modelN,out_path=out_path,use_float8 =False)
#model.keras_model.set_weights(model.quantize_weights_and_bias(model.get_weights())) 

# %% software simulation of data streamer  
#commands = model.translate_conv2d(model.keras_model.layers[2],0)  
commands = model.translate()
    
# %% hardware simulation of data streamer 
#sim_model.run_hw_sim(layer_outs[0][0][1:2,:,:,:], 1, hw_path, testbench='tb_data_streamer', max_in_channels=32, max_tiles = 1, waveform=True)
# %% software simulation   
