import os
import numpy as np
from elevate import elevate
from intuitus_nn import Framebuffer, Camera
import cv2

elevate()
cam = Camera('/dev/video0')
fb = Framebuffer('/dev/fb0')
status, img_yuv = cam.capture()
print(status)
print(img_yuv.shape)
img_bgr = cv2.cvtColor(img_yuv,cv2.COLOR_YUV2BGR_Y422)
img_bgr = cv2.flip(img_bgr, 0)
cv2.imwrite('cam_data/cam_out.png', img_bgr) 
print(np.sqrt(np.mean(img_bgr[...,0].astype(np.float32)**2)))
print(np.sqrt(np.mean(img_bgr[...,1].astype(np.float32)**2)))
print(np.sqrt(np.mean(img_bgr[...,2].astype(np.float32)**2)))
img_bgr[...,1] = np.uint8(img_bgr[...,1]*0.66)
print(np.sqrt(np.mean(img_bgr[...,1].astype(np.float32)**2)))

print(img_bgr.shape)
fb.show(img_bgr,0)
