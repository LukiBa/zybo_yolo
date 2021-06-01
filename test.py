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
layer_nbr = 6
iterations = 1

print_last = True
exectue_net = True 

test_img_npz = np.load(str(command_path / 'test_img_int8.npz'),allow_pickle=True)
test_img = test_img_npz['img'].astype(np.uint8).reshape((1,48,48))
#print(test_img.shape)
#print(test_img[0,...])

elevate()
Net = Intuitus_intf()
#Net.self_test()
Net = nn.Sequential(command_path)
buffer = Net.input(1,48,48)
buffer = Net.conv2d(buffer,32,(3,3),command_file = 'conv2d_int8.npz')
buffer = Net.conv2d(buffer,32,(3,3),command_file = 'conv2d_int8_1.npz')
buffer = Net.conv2d(buffer,32,(3,3),strides=(2,2),command_file = 'conv2d_int8_2.npz')
buffer = Net.conv2d(buffer,64,(3,3),command_file = 'conv2d_int8_3.npz')
buffer = Net.conv2d(buffer,64,(3,3),command_file = 'conv2d_int8_4.npz')
buffer = Net.conv2d(buffer,64,(3,3),strides=(2,2),command_file = 'conv2d_int8_5.npz')
#buffer = Net.conv2d(buffer,128,(3,3),command_file = 'conv2d_int8_6.npz')
#buffer = Net.conv2d(buffer,128,(3,3),command_file = 'conv2d_int8_7.npz')
#buffer = Net.conv2d(buffer,128,(3,3),strides=(2,2),command_file = 'conv2d_int8_8.npz')
#buffer = Net.upsample(buffer)
buffer = Net.output(buffer)


if print_last:
    Net.summary()
    Net.print_layer_dma_info(layer_nbr)
if exectue_net:
    for i in range(iterations):
        img_float, image = Net(test_img)
    outfile_name = 'fmap_out_' + str(layer_nbr) + '.npy'
    np.save(str(out_path / outfile_name), image)
    np.save(str(out_path / 'fmap_float_out.npy'), img_float)
    print(img_float.shape)
    print('done..')

# l1_conv2d_commands = np.load(str(command_path / 'conv2d_fl8.npz'),allow_pickle=True)
# l1_command_lengths = l1_conv2d_commands['tx_com_len'].astype(np.uint32)
# l1_command_block = l1_conv2d_commands['tx_bin'].astype(np.int32)
# l1_tile_tx_arr = l1_conv2d_commands['tx_tile'].astype(np.uint32)
# l1_tile_rx_arr = l1_conv2d_commands['rx_tile'].astype(np.uint32)
# Net.conv2d(1,4,0,1,48,48,32,int(l1_tile_rx_arr[0,6]),l1_tile_tx_arr,l1_tile_rx_arr[:,:6],l1_command_block,l1_command_lengths)

# if layer_nbr > 1:
#     l2_conv2d_commands = np.load(str(command_path / 'conv2d_fl8_1.npz'),allow_pickle=True)
#     l2_command_lengths = l2_conv2d_commands['tx_com_len'].astype(np.uint32)
#     l2_command_block = l2_conv2d_commands['tx_bin'].astype(np.int32)
#     l2_tile_tx_arr = l2_conv2d_commands['tx_tile'].astype(np.uint32)
#     l2_tile_rx_arr = l2_conv2d_commands['rx_tile'].astype(np.uint32)
#     Net.conv2d(2,4,1,32,48,48,32,int(l2_tile_rx_arr[0,6]),l2_tile_tx_arr,l2_tile_rx_arr[:,:6],l2_command_block,l2_command_lengths)

# if layer_nbr > 2:
#     l3_conv2d_commands = np.load(str(command_path / 'conv2d_fl8_2.npz'),allow_pickle=True)
#     l3_command_lengths = l3_conv2d_commands['tx_com_len'].astype(np.uint32)
#     l3_command_block = l3_conv2d_commands['tx_bin'].astype(np.int32)
#     l3_tile_tx_arr = l3_conv2d_commands['tx_tile'].astype(np.uint32)
#     l3_tile_rx_arr = l3_conv2d_commands['rx_tile'].astype(np.uint32)
#     Net.conv2d(3,4,2,32,24,24,32,int(l3_tile_rx_arr[0,6]),l3_tile_tx_arr,l3_tile_rx_arr[:,:6],l3_command_block,l3_command_lengths)

# if layer_nbr > 3:
#     l4_conv2d_commands = np.load(str(command_path / 'conv2d_fl8_3.npz'),allow_pickle=True)
#     l4_command_lengths = l4_conv2d_commands['tx_com_len'].astype(np.uint32)
#     l4_command_block = l4_conv2d_commands['tx_bin'].astype(np.int32)
#     l4_tile_tx_arr = l4_conv2d_commands['tx_tile'].astype(np.uint32)
#     l4_tile_rx_arr = l4_conv2d_commands['rx_tile'].astype(np.uint32)
#     Net.conv2d(4,4,3,32,24,24,64,int(l4_tile_rx_arr[0,6]),l4_tile_tx_arr,l4_tile_rx_arr[:,:6],l4_command_block,l4_command_lengths)

# if layer_nbr > 4:
#     l5_conv2d_commands = np.load(str(command_path / 'conv2d_fl8_4.npz'),allow_pickle=True)
#     l5_command_lengths = l5_conv2d_commands['tx_com_len'].astype(np.uint32)
#     l5_command_block = l5_conv2d_commands['tx_bin'].astype(np.int32)
#     l5_tile_tx_arr = l5_conv2d_commands['tx_tile'].astype(np.uint32)
#     l5_tile_rx_arr = l5_conv2d_commands['rx_tile'].astype(np.uint32)
#     Net.conv2d(5,4,4,64,24,24,64,int(l5_tile_rx_arr[0,6]),l5_tile_tx_arr,l5_tile_rx_arr[:,:6],l5_command_block,l5_command_lengths)

# if layer_nbr > 5:
#     l6_conv2d_commands = np.load(str(command_path / 'conv2d_fl8_5.npz'),allow_pickle=True)
#     l6_command_lengths = l6_conv2d_commands['tx_com_len'].astype(np.uint32)
#     l6_command_block = l6_conv2d_commands['tx_bin'].astype(np.int32)
#     l6_tile_tx_arr = l6_conv2d_commands['tx_tile'].astype(np.uint32)
#     l6_tile_rx_arr = l6_conv2d_commands['rx_tile'].astype(np.uint32)
#     Net.conv2d(6,4,5,64,12,12,64,int(l6_tile_rx_arr[0,6]),l6_tile_tx_arr,l6_tile_rx_arr[:,:6],l6_command_block,l6_command_lengths)

# if layer_nbr > 6:
#     l7_conv2d_commands = np.load(str(command_path / 'conv2d_fl8_6.npz'),allow_pickle=True)
#     l7_command_lengths = l7_conv2d_commands['tx_com_len'].astype(np.uint32)
#     l7_command_block = l7_conv2d_commands['tx_bin'].astype(np.int32)
#     l7_tile_tx_arr = l7_conv2d_commands['tx_tile'].astype(np.uint32)
#     l7_tile_rx_arr = l7_conv2d_commands['rx_tile'].astype(np.uint32)
#     Net.conv2d(7,4,6,64,12,12,128,int(l7_tile_rx_arr[0,6]),l7_tile_tx_arr,l7_tile_rx_arr[:,:6],l7_command_block,l7_command_lengths)

# if layer_nbr > 7:
#     l8_conv2d_commands = np.load(str(command_path / 'conv2d_fl8_7.npz'),allow_pickle=True)
#     l8_command_lengths = l8_conv2d_commands['tx_com_len'].astype(np.uint32)
#     l8_command_block = l8_conv2d_commands['tx_bin'].astype(np.int32)
#     l8_tile_tx_arr = l8_conv2d_commands['tx_tile'].astype(np.uint32)
#     l8_tile_rx_arr = l8_conv2d_commands['rx_tile'].astype(np.uint32)
#     Net.conv2d(8,4,7,128,12,12,128,int(l8_tile_rx_arr[0,6]),l8_tile_tx_arr,l8_tile_rx_arr[:,:6],l8_command_block,l8_command_lengths)

# if layer_nbr > 8:
#     l9_conv2d_commands = np.load(str(command_path / 'conv2d_fl8_8.npz'),allow_pickle=True)
#     l9_command_lengths = l9_conv2d_commands['tx_com_len'].astype(np.uint32)
#     l9_command_block = l9_conv2d_commands['tx_bin'].astype(np.int32)
#     l9_tile_tx_arr = l9_conv2d_commands['tx_tile'].astype(np.uint32)
#     l9_tile_rx_arr = l9_conv2d_commands['rx_tile'].astype(np.uint32)
#     Net.conv2d(9,4,8,128,6,6,128,int(l9_tile_rx_arr[0,6]),l9_tile_tx_arr,l9_tile_rx_arr[:,:6],l9_command_block,l9_command_lengths)

# Net.print_network()
# if print_last:
#     Net.print_layer(layer_nbr)
# if exectue_net:
#     status, image = Net.execute(test_img)
#     print(status)
#     status, img_float = Net.float8_to_float32(image)
#     print(status)
#     outfile_name = 'fmap_out_' + str(layer_nbr) + '.npy'
#     np.save(str(out_path / outfile_name), image)
#     np.save(str(out_path / 'fmap_float_out.npy'), img_float)
