import os
import numpy as np
from elevate import elevate
from intuitus_intf import Framebuffer
import cv2

elevate(graphical=False)

screen_width = 1440
screen_height = 900

fb = Framebuffer('/dev/fb0')

black_screen = np.zeros([screen_height,screen_width,3],dtype=np.uint8)
fb.show(black_screen,0)

img_rgb = cv2.imread('cam_data/dog.jpg') 
print(img_rgb.shape)
img_bgr = cv2.cvtColor(img_rgb,cv2.COLOR_RGB2BGR)

print(img_bgr.shape)
fb.show(img_rgb,(1920*20+500)*3)