# -*- coding: utf-8 -*-
"""
Created on Sun Feb 21 11:40:32 2021

@author: lukas
"""

#%% import public modules
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
#import Intuitus.util.optain_dataset as load 
from Intuitus.models import Intuitus_Model
from Intuitus.layers import Conv2D_fl8
from Intuitus.core import float8, float6, MACC, float12
import IntuitusExtension as C_impl
#%% Parameter 
MODEL_NAME = './checkpoints/yolov4-416-tiny'
json_root = pathlib.Path(Intuitus.__file__).absolute().parents[0] / "Intuitus_HyperPar.json"
model_path = pathlib.Path(__file__).absolute().parent / MODEL_NAME
hw_path = pathlib.Path(__file__).absolute().parents[4] / "Hardware" 

# %% load pretrained model. Use example_nn.py for training the model
try: modelN = keras_models.load_model(str(model_path))
except:
    Exception('Create the model first!')
    
# %% Initialize Intuitus model using pretrained keras model
model = Intuitus_Model(modelN,json_root)
model.keras_model.set_weights(model.quantize_weights_and_bias(model.get_weights())) 
plot_model(modelN, to_file='{}/model.png'.format(MODEL_NAME), show_shapes=True, show_layer_names=True)

# %% software simulation of data streamer  
commands = model.translate_conv2d(model.keras_model.layers[2],0)  
    
# %% hardware simulation of data streamer 
#sim_model.run_hw_sim(layer_outs[0][0][1:2,:,:,:], 1, hw_path, testbench='tb_data_streamer', max_in_channels=32, max_tiles = 1, waveform=True)
# %% software simulation   
