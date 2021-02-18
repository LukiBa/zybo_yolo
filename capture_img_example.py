import os
import numpy as np
from elevate import elevate
from intuitus_intf import Framebuffer, Camera
import cv2

cam = Camera('/dev/video0')
fb = Framebuffer('/dev/fb0')
status, img_yuv = cam.capture()
print(status)
print(img_yuv.shape)
img_bgr = cv2.cvtColor(img_yuv,cv2.COLOR_YUV2BGR_Y422)
cv2.imwrite('cam_data/cam_out.png', img_bgr) 

print(img_bgr.shape)
fb.update(img_bgr)
