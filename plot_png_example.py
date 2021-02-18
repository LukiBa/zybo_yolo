import os
import numpy as np
from elevate import elevate
from intuitus_intf import Framebuffer
import cv2

fb = Framebuffer('/dev/fb0')

black_screen = np.zeros([1080,1920,3],dtype=np.uint8)
fb.update(black_screen,0)

img_rgb = cv2.imread('cam_data/dog.jpg') 
print(img_rgb.shape)
img_bgr = cv2.cvtColor(img_rgb,cv2.COLOR_RGB2BGR)

print(img_bgr.shape)
fb.update(img_rgb,(1920*20+500)*3)
