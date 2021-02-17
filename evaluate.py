import os
import numpy as np
from elevate import elevate
from intuitus_intf import Intuitus_intf

try: elevate()
except: Exception(OSError,"Failed to get root previleges.")
network = Intuitus_intf()

try: network.self_test()
except: Exception(OSError,"Self test failed.")

try: network.input_layer(48,48,1)
except: RuntimeError

tiles = np.zeros([4,8])
tile[0,:] = np.array([0,31,0,37,0,30,0,36])
tile[1,:] = np.array([30,48,0,37,30,48,0,36])
tile[2,:] = np.array([0,31,36,48,0,30,36,48])
tile[3,:] = np.array([30,48,36,48,30,48,36,48])

try: network.conv2d(1,1,0,1,48,48,32,tiles,)
except: RuntimeError

