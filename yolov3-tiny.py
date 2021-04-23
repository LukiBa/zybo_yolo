import numpy as np
import cv2
import pathlib
import argparse

from intuitus_intf import Intuitus_intf, Framebuffer, Camera
import core.wrapper as nn 


def _create_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--command_path',  type=str, default='./intuitus_commands/yolov3-tiny-commands', help='path to intuitus command files')
    parser.add_argument('--image',  type=str, default='./cam_data/dog.jpg', help='path to input image')
    parser.add_argument('--output',  type=str, default='./fmap_out', help='path to output')
    parser.add_argument('--tiny', type=bool, default=True, help='is yolo-tiny or not')
    parser.add_argument('--size', type=int, default=320, help='define input size of export model')
    parser.add_argument('--iou', type=float, default=0.45, help='iou threshold')
    parser.add_argument('--score', type=float, default=0.25, help='score threshold')
    parser.add_argument('--use_cam', type=bool, default=False, help='use camera for input img')
    parser.add_argument('--use_fb', type=bool, default=False, help='write output to framebuffer')
    return parser.parse_args() 

print_last = True
exectue_net = False 

def main(flags):
    input_size = flags.size
    image_path = flags.image
    command_path = pathlib.Path(flags.command_path).absolute()
    out_path = pathlib.Path(flags.output).absolute()

    if flags.use_cam:
        cam = Camera('/dev/video0')
        status, original_image = cam.capture()
        original_image = cv2.cvtColor(original_image,cv2.COLOR_YUV2BGR_Y422)
        original_image = cv2.flip(original_image, 0)
    else:
        original_image = cv2.imread(image_path)

    image_data = cv2.resize(original_image, (input_size, input_size))
    b,g,r = cv2.split(image_data)
    image_data = np.stack([r,g,b])
    image_data = image_data / 255.

    Net = nn.Sequential(command_path)
    buffer = Net.Input(3,input_size,input_size)

    buffer = Net.conv2d(buffer,16,(3,3),max_pooling=True,command_file='conv2d_0.npz')
    buffer = Net.conv2d(buffer,32,(3,3),max_pooling=True,command_file='conv2d_1.npz')
    buffer = Net.conv2d(buffer,64,(3,3),max_pooling=True,command_file='conv2d_2.npz')
    buffer = Net.conv2d(buffer,128,(3,3),max_pooling=True,command_file='conv2d_3.npz')
    buffer = Net.conv2d(buffer,256,(3,3),max_pooling=True,command_file='conv2d_4.npz')
    buffer = Net.conv2d(buffer,512,(3,3),command_file='conv2d_5.npz')


    

    Net.summary()
    if print_last:
        Net.print_layer_dma_info(layer_nbr)
    if exectue_net:    
        img_fl32, img_fl8 = Net(test_img)
        outfile_name = 'fmap_out_' + str(layer_nbr) + '.npy'
        np.save(str(out_path / outfile_name), img_fl8)
        np.save(str(out_path / 'fmap_float_out.npy'), img_fl32)

if __name__ == '__main__':
    flags = _create_parser()
    main(flags)
