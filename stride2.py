import numpy as np
import cv2
import pathlib
from elevate import elevate
from intuitus_intf import Intuitus_intf, Framebuffer

command_path = pathlib.Path(__file__).absolute().parent / 'checkpoints' / 'command_out'
out_path = pathlib.Path(__file__).absolute().parent / 'output' 
input_size = 48
layer_nbr = 1
print_last = False
exectue_net = True 

test_img_npz = np.load(str(command_path / 'test_img.npz'),allow_pickle=True)
test_img = test_img_npz['img'].astype(np.uint8)
print(test_img.shape)

elevate()
Net = Intuitus_intf()
#Net.self_test()
Net.input_layer(1,48,48)

l1_conv2d_commands = np.load(str(command_path / 'stride2_1.npz'),allow_pickle=True)
l1_command_lengths = l1_conv2d_commands['tx_com_len'].astype(np.uint32)
l1_command_block = l1_conv2d_commands['tx_bin'].astype(np.int32)
l1_tile_tx_arr = l1_conv2d_commands['tx_tile'].astype(np.uint32)
l1_tile_rx_arr = l1_conv2d_commands['rx_tile'].astype(np.uint32)

Net.conv2d(1,4,0,1,24,24,32,l1_tile_tx_arr,l1_tile_rx_arr[:,2:],l1_command_block,l1_command_lengths)


Net.print_network()
Net.print_layer(1)
if exectue_net:
    status, image = Net.execute(test_img)
    print(image[0,...])
    status, img_float = Net.float8_to_float32(image)
    np.save(str(out_path / 'fmap_out.npy'), image)
    np.save(str(out_path / 'fmap_float_out.npy'), img_float)
