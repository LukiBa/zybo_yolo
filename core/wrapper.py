import numpy as np
import pathlib
from intuitus_intf import Intuitus_intf
from elevate import elevate

class buffer:
    def __init__(self,id,channel,height,width):
        self.id = id 
        self._channel = channel
        self._height = height 
        self._width = width
        self.size = self._channel * self._height * self._width
        self.shape = (self._channel,self._height,self._width)

    def __len__(self):
        return self.size

    @property
    def channel(self):
        return self._channel
    @channel.setter
    def channel(self,channel):
        self._channel = channel
        self.size = self._channel * self._height * self._width
        self.shape = (self._channel,self._height,self._width)

    @property
    def height(self):
        return self._height
    @height.setter
    def height(self,height):    
        self._height = height
        self.size = self._channel * self._height * self._width
        self.shape = (self._channel,self._height,self._width)

    @property
    def width(self):
        return self._width 
    @width.setter
    def width(self,width):
        self._width = width
        self.size = self._channel * self._height * self._width 
        self.shape = (self._channel,self._height,self._width)       


class Sequential:
    def __init__(self,command_path):
        self.layer_types = {'Input'             : 0,
                            'Output'            : 1,
                            'Conv1x1'           : 2,
                            'InvBottleneck3x3'  : 3,
                            'InvBottleneck5x5'  : 4,
                            'Conv3x3'           : 5,
                            'Conv5x5'           : 6,
                            'Residual'          : 7,
                            'Concat'            : 8,
                            'Test_loop'         : 9}
        
        elevate() # get root privileges. Required to open kernel driver 
        self.command_path = command_path
        self.Net = Intuitus_intf()
        self.layer_nbr = 0
        self.has_input = False
        self.has_output = False    
        self.outputs = []
    def __len__(self):
        return self.layer_nbr

    def __call__(self,input):
        if not self.has_input or not self.has_output:
            raise Exception("network requires input and output layer") 
        status, fmap = self.Net.execute(input)
        if status != 0:
            raise Exception("error in execution of network")   
    	
        if len(self.outputs) == 1:
            out = fmap.reshape(self.outputs[0].shape)
            status, img_float = self.Net.float8_to_float32(out)
            if status != 0:
                raise Exception("error converting float8 to float32")  
            return img_float, out

        outpos = 0
        out_fmaps = []
        out_float = []
        for outs in self.outputs:
            out = fmap[outpos:outputs.size].reshape(outs.shape)
            outpos += outputs.size
            status, img_float = self.Net.float8_to_float32(out)
            if status != 0:
                raise Exception("error converting float8 to float32") 
            out_fmaps.append(out)
            out_float.append(img_float)    
        return out_float, out_fmaps

    def forward_layer(self,layer_id,input):
        status, image = self.Net.execute_layer(layer_id,input)
        if status != 0:
            raise Exception("error in execution of layer {}".format(layer_id))        
        status, img_float = self.Net.float8_to_float32(image)
        if status != 0:
            raise Exception("error converting float8 to float32") 
        return img_float, image

    def input(self,channel,height,width):
        status=  self.Net.input_layer(channel,height,width)
        if status != 0:
            raise Exception("error configuring input layer")
        self.has_input = True       
        return buffer(0,channel,height,width)

    def output(self,in_buffer):
        self.layer_nbr += 1
        status=  self.Net.output_layer(self.layer_nbr,in_buffer.id)
        if status != 0:
            self.layer_nbr -= 1
            raise Exception("error configuring output layer @{}".format(self.layer_nbr+1))
            
        self.has_output = True 
        out_buffer = buffer(self.layer_nbr,in_buffer.channel,in_buffer.height,in_buffer.width)
        self.outputs.append(out_buffer)
        return out_buffer

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
            out_height = int(out_height/2)
            out_width = int(out_width/2)

        status = self.Net.conv2d(self.layer_nbr,layer_type,in_buffer.id,in_buffer.channel,out_height,out_width,filters,int(tile_rx_arr[0,6]),tile_tx_arr,tile_rx_arr[:,:6],command_block,command_lengths)
        if status != 0:
            self.layer_nbr -= 1
            raise Exception("error configuring network. Conv2d layer with id: {}".format(self.layer_nbr))

        return buffer(self.layer_nbr,filters,out_height,out_width)

    def maxpool2d(self,in_buffer):
        self.layer_nbr += 1
        status=  self.Net.maxpool2d(self.layer_nbr,in_buffer.id)
        if status != 0:
            self.layer_nbr -= 1
            raise Exception("error configuring maxpool2d layer @{}".format(self.layer_nbr+1))
        
        out_height = int(in_buffer.height/2)
        out_width = int(in_buffer.width/2)

        return buffer(self.layer_nbr,in_buffer.channel,out_height,out_width)

    def upsample(self,in_buffer):
        self.layer_nbr += 1
        out_height = int(in_buffer.height*2)
        out_width = int(in_buffer.width*2)        
        status=  self.Net.upsample(self.layer_nbr,in_buffer.id,in_buffer.channel,out_height, out_width)
        if status != 0:
            self.layer_nbr -= 1
            raise Exception("error configuring upsample layer @{}".format(self.layer_nbr+1))


        return buffer(self.layer_nbr,in_buffer.channel,out_height,out_width)

    def concat(self, in_buffer_0, in_buffer_1):
        self.layer_nbr += 1
        if in_buffer_0.height != in_buffer_1.height or in_buffer_0.width != in_buffer_0.width:
            self.layer_nbr -= 1
            raise Exception("error configuring network. In Concat layer {}. Width and Height of buffer: {} and buffer: {} have to be equal for concatenation.".format(self.layer_nbr,in_buffer_0.id,in_buffer_1.id))

        status = self.Net.concat(self.layer_nbr,in_buffer_0.id,in_buffer_1.id)
        if status != 0:
            self.layer_nbr -= 1
            raise Exception("error configuring network. Concat layer with id: {}".format(self.layer_nbr))        

        return buffer(self.layer_nbr,in_buffer_0.filters+in_buffer_1+filters,in_buffer_0.height,in_buffer_0.width)

    def split(self, in_buffer, groups):
        self.layer_nbr += 1
        status = self.Net.split(self.layer_nbr,in_buffer.id,groups)
        if status != 0:
            self.layer_nbr -= 1
            raise Exception("error configuring network. Split layer with id: {}. Failed to split buffer {} into {} groups.".format(self.layer_nbr,in_buffer.id,groups))        

        out_buffers = []
        for i in range(groups):
            out_buffers = out_buffers.append(buffer(self.layer_nbr+i,int(in_buffer.channel/groups),in_buffer.height,in_buffer.width))
        
        self.layer_nbr += groups-1
        return tuple(out_buffers)

    def summary(self):
        self.Net.print_network()
    def print_layer_dma_info(self,layer_nbr):
        self.Net.print_layer(layer_nbr)