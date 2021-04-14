import numpy as np
import pathlib
from intuitus_intf import Intuitus_intf
from elevate import elevate

class buffer:
    def __init__(self,id,channel,height,width):
        self.id = id 
        self.channel = channel
        self.height = height 
        self.width = width


class Sequential:
    def __init__(self,command_path):
        self.layer_types = {'Input'             : 0,
                            'Conv1x1'           : 1,
                            'InvBottleneck3x3'  : 2,
                            'InvBottleneck5x5'  : 3,
                            'Conv3x3'           : 4,
                            'Conv5x5'           : 5,
                            'Residual'          : 6,
                            'Concat'            : 7,
                            'Test_loop'         : 8}
        elevate()
        self.Net = Intuitus_intf()
        self.layer_nbr = 0
        self.command_path = command_path
    def __call__(self,input):
        status, image = self.Net.execute(input)
        if status != 0:
            raise Exception("error in execution of network")        
        status, img_float = Net.float8_to_float32(image)
        if status != 0:
            raise Exception("error converting float8 to float32") 
        return img_float, image

    def Input(self,channel,height,width):
        status=  self.Net.input_layer(channel,height,width)
        if status != 0:
            raise Exception("error configuring input layer")
        self.layer_nbr += 1
        return buffer(0,channel,height,width)

    def conv2d(self, in_buffer,filters,kernel_size,strides = (1,1),max_pooling=False, command_file = None):
        self.layer_nbr += 1
        if command_file == None:
            command_file = self.command_path / "conv2d_{}".format(self.layer_nbr)
        else:
            command_file = self.command_path / command_file

        conv2d_commands = np.load(command_file,allow_pickle=True)
        command_lengths = conv2d_commands['tx_com_len'].astype(np.uint32)
        command_block = conv2d_commands['tx_bin'].astype(np.int32)
        tile_tx_arr = conv2d_commands['tx_tile'].astype(np.uint32)
        tile_rx_arr = conv2d_commands['rx_tile'].astype(np.uint32)
        if kernel_size == (1,1):
            layer_type = self.layer_types['Conv1x1']
        elif kernel_size == (3,3):
            layer_type = self.layer_types['Conv3x3']
        elif kernel_size == (5,5):
            layer_type = self.layer_types['Conv5x5']
        else:
            self.layer_nbr -= 1
            raise NotImplementedError("Used kernel size is not supported. Use: (1,1), (3,3) or (5,5)")
        
        if strides == (1,1):
            stride = 1
        elif strides == (2,2):
            stride = 2
        else:
            self.layer_nbr -= 1
            raise NotImplementedError("Used strides are not supported. Use: (1,1) or (2,2)")

        out_height = in_buffer.height
        out_width = in_buffer.width
        if max_pooling or stride == 2:
            out_height /= 2
            out_width /= 2

        status = self.Net.conv2d(self.layer_nbr,layer_type,in_buffer.id,in_buffer.channel,out_height,out_width,filters,tile_rx_arr[0,6],tile_tx_arr,tile_rx_arr[:,:4],command_block,command_lengths)
        if status != 0:
            self.layer_nbr -= 1
            raise Exception("error configuring network. Layer {}".format(self.layer_nbr))

        return buffer(self.layer_nbr,filters,out_height,out_width)
    def summary(self):
        self.Net.print_network()
    def print_layer_dma_info(self,layer_nbr):
        self.Net.print_layer(layer_nbr)