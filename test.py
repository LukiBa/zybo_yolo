import numpy as np
import cv2
import pathlib
from elevate import elevate
from intuitus_intf import Intuitus_intf, Framebuffer
import core.wrapper as nn 

command_path = pathlib.Path(__file__).absolute().parent / 'checkpoints' / 'command_out'
out_path = pathlib.Path(__file__).absolute().parent / 'output' 
out_path.mkdir(parents=True,exist_ok=True)
input_size = 48
layer_nbr = 9
iterations = 1000

print_last = True
exectue_net = True 

test_img_npz = np.load(str(command_path / 'test_img_int8.npz'),allow_pickle=True)
test_img = test_img_npz['img'].astype(np.uint8).reshape((1,48,48))
#print(test_img.shape)
#print(test_img[0,...])

elevate()
Net = nn.Sequential(command_path)
buffer = Net.input(1,48,48)
buffer = Net.conv2d(buffer,32,(3,3),command_file = 'conv2d_int8.npz')
buffer = Net.conv2d(buffer,32,(3,3),command_file = 'conv2d_int8_1.npz')
buffer = Net.conv2d(buffer,32,(3,3),strides=(2,2),command_file = 'conv2d_int8_2.npz')
buffer = Net.conv2d(buffer,64,(3,3),command_file = 'conv2d_int8_3.npz')
buffer = Net.conv2d(buffer,64,(3,3),command_file = 'conv2d_int8_4.npz')
buffer = Net.conv2d(buffer,64,(3,3),strides=(2,2),command_file = 'conv2d_int8_5.npz')
buffer = Net.conv2d(buffer,128,(3,3),command_file = 'conv2d_int8_6.npz')
buffer = Net.conv2d(buffer,128,(3,3),command_file = 'conv2d_int8_7.npz')
buffer = Net.conv2d(buffer,128,(3,3),strides=(2,2),command_file = 'conv2d_int8_8.npz')
#buffer = Net.upsample(buffer)
buffer = Net.output(buffer)


if print_last:
    Net.summary()
    Net.print_layer_dma_info(layer_nbr)
if exectue_net:
    for i in range(iterations):
        image = Net(test_img)
    outfile_name = 'fmap_out_' + str(layer_nbr) + '.npy'
    np.save(str(out_path / outfile_name), image)
    print('done..')