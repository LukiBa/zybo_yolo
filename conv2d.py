import numpy as np
import cv2
import pathlib
from elevate import elevate
from intuitus_intf import Intuitus_intf, Framebuffer
import time

screen_width = 1440
screen_height = 900

command_path = pathlib.Path(__file__).absolute().parent / 'checkpoints' / 'output'
conv2d_commands = np.load(str(command_path / 'conv2d.npz'),allow_pickle=True)

command_lengths = conv2d_commands['tx_com_len'].astype(np.uint32)
command_block = conv2d_commands['tx_bin'].astype(np.int32)
tile_tx_arr = conv2d_commands['tx_tile'].astype(np.uint32)
tile_rx_arr = conv2d_commands['rx_tile'].astype(np.uint32)

input_size = 416
image_path = './cam_data/dog.jpg'
original_image = cv2.imread(image_path)
original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
image_data = cv2.resize(original_image, (input_size, input_size))
image_data = image_data.astype(np.uint8)
b, g ,r = cv2.split(image_data)
new_img = np.array([r,g,b])
#image_data = np.ones([3,input_size,input_size])
#for i in range(input_size):
#    image_data[:,:,i] *= i 

#image_data = image_data.astype(np.uint8)

#print(tile_tx_arr)
#print(tile_rx_arr)

elevate(graphical=True)
Net = Intuitus_intf()
#Net.self_test()
Net.input_layer(3,416,416)
Net.conv2d(1,4,0,3,416,416,3,tile_rx_arr[:,2:],tile_rx_arr[:,2:],command_block,command_lengths)
#Net.conv2d(2,4,1,3,416,416,3,tile_rx_arr[:,2:],tile_rx_arr,command_block,command_lengths)
#Net.print_network()
#Net.print_layer(1)
print(new_img.shape)
status, image = Net.execute_layer(1,new_img)
print(status)
print(image.shape)
print_img = cv2.merge((image[2,:,:],image[1,:,:],image[0,:,:]))
print(print_img.shape)
cv2.imwrite('cam_data/test.png', print_img) 
#print(image[:10,:10])
print("Execution status: {}".format(status))
fb = Framebuffer('/dev/fb0')
black_screen = np.zeros([screen_height,screen_width,3],dtype=np.uint8)
fb.show(black_screen,0)
img_bgr = cv2.cvtColor(print_img,cv2.COLOR_RGB2BGR)
fb.show(img_bgr,0) # (1920*20+500)*3
