import numpy as np
import cv2
import pathlib

from intuitus_intf import Intuitus_intf, Framebuffer
import core.wrapper as nn 

command_path = pathlib.Path(__file__).absolute().parent / 'checkpoints' / 'yolov4-tiny' / 'command_out'
out_path = pathlib.Path(__file__).absolute().parent / 'output' 
input_size = 416
layer_nbr = 1

print_last = True
exectue_net = True 

test_img_npz = np.load(str(command_path / 'test_img.npz'),allow_pickle=True)
test_img = test_img_npz['img'].astype(np.uint8)
print(test_img.shape)

Net = nn.Sequential(command_path)
buffer = Net.Input(3,416,416)
buffer = Net.conv2d(buffer,32,(3,3),command_file = 'conv2d_fl8.npz')


Net.summary()
if print_last:
    Net.print_layer_dma_info(layer_nbr)
if exectue_net:    
    img_fl32, img_fl8 = Net(test_img)
    outfile_name = 'fmap_out_' + str(layer_nbr) + '.npy'
    np.save(str(out_path / outfile_name), img_fl8)
    np.save(str(out_path / 'fmap_float_out.npy'), img_fl32)
