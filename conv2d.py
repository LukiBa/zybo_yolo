import numpy as np
import pathlib
from elevate import elevate
from intuitus_intf import Intuitus_intf

command_path = pathlib.Path(__file__).absolute().parent / 'checkpoints' / 'output'

conv2d_commands = np.load(str(command_path / 'conv2d.npz'),allow_pickle=True)
print(len(conv2d_commands['rx']))

elevate()
Net = Intuitus_intf()
#Net.self_test()
Net.input_layer(416,416,3)
Net.conv2d(1,4,0,3,416,416,16,)
Net.print_network()
