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

#%% import custom modules 
import Intuitus
from Intuitus.models import Intuitus_Model
from Intuitus.layers import Conv2D_fl8
from Intuitus.core import float8, float6, MACC, float12
import IntuitusExtension as C_impl
#%% Parameter 
parser = argparse.ArgumentParser()
parser.add_argument('--model_path',  type=str, default='./checkpoints/yolov3-tiny-416-backbone', help='path to model')
parser.add_argument('--out_path',  type=str, default='./intuitus_commands/yolov3-tiny-commands', help='command output path')
flags=parser.parse_args()

hw_path = pathlib.Path(Intuitus.__file__).absolute().parents[3]  / "Hardware" 
model_path = pathlib.Path(flags.model_path).absolute()
out_path = pathlib.Path(flags.out_path).absolute()
# %% load pretrained model. Use example_nn.py for training the model
modelN = keras_models.load_model(str(model_path))
modelN.summary()
    
# %% Initialize Intuitus model using pretrained keras model
model = Intuitus_Model(modelN,out_path=out_path)
#model.keras_model.set_weights(model.quantize_weights_and_bias(model.get_weights())) 

# %% software simulation of data streamer  
#commands = model.translate_conv2d(model.keras_model.layers[2],0)  
commands = model.translate()
    
# %% hardware simulation of data streamer 
#sim_model.run_hw_sim(layer_outs[0][0][1:2,:,:,:], 1, hw_path, testbench='tb_data_streamer', max_in_channels=32, max_tiles = 1, waveform=True)
# %% software simulation   
