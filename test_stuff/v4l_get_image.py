# -*- coding: utf-8 -*-
"""
Created on Tue Nov 17 16:02:55 2020

@author: lukas
"""
#!/usr/bin/env python3

from v4l2 import *
import fcntl
import mmap
import select
import time
import os 

#def set_format
#os.system(media-ctl -d /dev/media0 -V '"ov5640 2-003c":0 [fmt:UYVY/'1920x1080'@1/'15' field:none]'')

vd = open('/dev/video0', 'rb+', buffering=0)


print(">> get device capabilities")
cp = v4l2_capability()
fcntl.ioctl(vd, VIDIOC_QUERYCAP, cp)

print("Driver:", "".join((chr(c) for c in cp.driver)))
print("Name:", "".join((chr(c) for c in cp.card)))
print("Is a video capture device?", bool(cp.capabilities & V4L2_CAP_VIDEO_CAPTURE))
print("Supports read() call?", bool(cp.capabilities &  V4L2_CAP_READWRITE))
print("Supports streaming?", bool(cp.capabilities & V4L2_CAP_STREAMING))
#%%
print(">> device setup")
fmt = v4l2_format()
fmt.type = V4L2_BUF_TYPE_VIDEO_CAPTURE
fcntl.ioctl(vd, VIDIOC_G_FMT, fmt)  # get current settings
print("width:", fmt.fmt.pix.width, "height", fmt.fmt.pix.height)
print("pxfmt:", "V4L2_PIX_FMT_YUYV" if fmt.fmt.pix.pixelformat == V4L2_PIX_FMT_YUYV else fmt.fmt.pix.pixelformat)
print("bytesperline:", fmt.fmt.pix.bytesperline)
print("sizeimage:", fmt.fmt.pix.sizeimage)
fcntl.ioctl(vd, VIDIOC_S_FMT, fmt)  # set whatever default settings we got before
#%%
print(">>> streamparam")  ## somewhere in here you can set the camera framerate
parm = v4l2_streamparm()
parm.type = V4L2_BUF_TYPE_VIDEO_CAPTURE
parm.parm.capture.capability = V4L2_CAP_TIMEPERFRAME
fcntl.ioctl(vd, VIDIOC_G_PARM, parm)
fcntl.ioctl(vd, VIDIOC_S_PARM, parm)  # just got with the defaults
#%%
print(">> init mmap capture")
req = v4l2_requestbuffers()
req.type = V4L2_BUF_TYPE_VIDEO_CAPTURE
req.memory = V4L2_MEMORY_MMAP
req.count = 1  # nr of buffer frames
fcntl.ioctl(vd, VIDIOC_REQBUFS, req)  # tell the driver that we want some buffers 
print("req.count", req.count)

#%%
buffers = []

print(">>> VIDIOC_QUERYBUF, mmap, VIDIOC_QBUF")
for ind in range(req.count):
    # setup a buffer
    buf = v4l2_buffer()
    buf.type = V4L2_BUF_TYPE_VIDEO_CAPTURE
    buf.memory = V4L2_MEMORY_MMAP
    buf.index = ind
    fcntl.ioctl(vd, VIDIOC_QUERYBUF, buf)

    mm = mmap.mmap(vd.fileno(), buf.length, mmap.MAP_SHARED, mmap.PROT_READ | mmap.PROT_WRITE, offset=buf.m.offset)
    buffers.append(mm)

    # queue the buffer for capture
    fcntl.ioctl(vd, VIDIOC_QBUF, buf)

#%%
print(">> Start streaming")
buf_type = v4l2_buf_type(V4L2_BUF_TYPE_VIDEO_CAPTURE)
fcntl.ioctl(vd, VIDIOC_STREAMON, buf_type)

#%%
print(">> Capture image")
t0 = time.time()
max_t = 1
ready_to_read, ready_to_write, in_error = ([], [], [])
print(">>> select")
while len(ready_to_read) == 0 and time.time() - t0 < max_t:
    ready_to_read, ready_to_write, in_error = select.select([vd], [], [], max_t)

#%%
print(">>> download buffers")
vid = open("video.yuv", "wb")

for _ in range(50):  # capture 50 frames
    buf = v4l2_buffer()
    buf.type = V4L2_BUF_TYPE_VIDEO_CAPTURE
    buf.memory = V4L2_MEMORY_MMAP
    fcntl.ioctl(vd, VIDIOC_DQBUF, buf)  # get image from the driver queue
    #print("buf.index", buf.index)
    mm = buffers[buf.index]
    # print first few pixels in gray scale part of yuvv format packed data
    print(" ".join(("{0:08b}".format(mm[x]) for x in range(0,16,2))))
    vid.write(mm.read())  # write the raw yuyv data from the buffer to the file
    #vid.write(bytes((bit for i, bit in enumerate(mm.read()) if not i % 2)))  # convert yuyv to grayscale
    mm.seek(0)
    fcntl.ioctl(vd, VIDIOC_QBUF, buf)  # requeue the buffer

#%%
print(">> Stop streaming")
fcntl.ioctl(vd, VIDIOC_STREAMOFF, buf_type)
vid.close()
vd.close()

print("video saved to video.yuv")
print("play it with mpv video.yuv --demuxer=rawvideo --demuxer-rawvideo-w=640 --demuxer-rawvideo-h=480 --demuxer-rawvideo-format=YUY2")