import numpy as np
import cv2
import pathlib
from elevate import elevate
from intuitus_intf import Intuitus_intf, Framebuffer

command_path = pathlib.Path(__file__).absolute().parent / 'checkpoints' / 'command_out'
out_path = pathlib.Path(__file__).absolute().parent / 'output' 
layer_nbr = 1
print_last = False
exectue_net = True 

test_img = np.load(str(out_path / 'fmap_out_6.npy')).astype(np.uint8)
print(test_img.shape)

elevate()
Net = Intuitus_intf()
#Net.self_test()
Net.input_layer(128,12,12)

l9_conv2d_commands = np.load(str(command_path / 'conv2d_fl8_8.npz'),allow_pickle=True)
l9_command_lengths = l9_conv2d_commands['tx_com_len'].astype(np.uint32)
l9_command_block = l9_conv2d_commands['tx_bin'].astype(np.int32)
l9_tile_tx_arr = l9_conv2d_commands['tx_tile'].astype(np.uint32)
l9_tile_rx_arr = l9_conv2d_commands['rx_tile'].astype(np.uint32)
Net.conv2d(1,4,0,128,6,6,128,l9_tile_tx_arr,l9_tile_rx_arr[:,2:],l9_command_block,l9_command_lengths)

Net.print_network()
Net.print_layer(1)
if exectue_net:
    status, image = Net.execute(test_img)
    print(image[0,...])
    status, img_float = Net.float8_to_float32(image)
    np.save(str(out_path / 'fmap_out_9.npy'), image)
    np.save(str(out_path / 'fmap_float_out.npy'), img_float)
