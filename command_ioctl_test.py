from elevate import elevate
from intuitus_intf import Intuitus_intf
import fcntl

elevate()
f = open('/dev/intuitus_vdma')
fcntl.ioctl(f,3,0)

