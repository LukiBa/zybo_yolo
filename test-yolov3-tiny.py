import numpy as np
import cv2
import pathlib
import argparse
import shutil

from intuitus_intf import Intuitus_intf, Framebuffer, Camera
import core.wrapper as nn 


def _create_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--command_path',  type=str, default='./intuitus_commands/yolov3-tiny-commands', help='path to intuitus command files')
    parser.add_argument('--image',  type=str, default='./cam_data/room384.npz', help='path to input image') # './cam_data/fmap384_out_4.npy') #
    parser.add_argument('--output',  type=str, default='./fmap_out', help='path to output')
    parser.add_argument('--tiny', type=bool, default=True, help='is yolo-tiny or not')
    parser.add_argument('--size', type=int, default=384, help='define input size of export model')
    parser.add_argument('--iou', type=float, default=0.45, help='iou threshold')
    parser.add_argument('--score', type=float, default=0.25, help='score threshold')
    parser.add_argument('--use_cam', type=bool, default=False, help='use camera for input img')
    parser.add_argument('--use_fb', type=bool, default=False, help='write output to framebuffer')
    parser.add_argument('--print', action='store_true', help='print network')
    parser.add_argument('--not_execute', action='store_true', help='print network')
    parser.add_argument('--iterations', type=int, default=1, help='execution iterations')
    return parser.parse_args() 


layer_nbr = 1

def main(flags):
    print(flags)
    #print(flags.print)
    print_last = flags.print
    exectue_net = not flags.not_execute 
    input_size = flags.size
    image_path = pathlib.Path(__file__).absolute().parent / flags.image
    command_path = pathlib.Path(__file__).absolute().parent / flags.command_path
    out_path = pathlib.Path(__file__).absolute().parent / flags.output

    if flags.use_cam:
        cam = Camera('/dev/video0')
        status, original_image = cam.capture()
        original_image = cv2.cvtColor(original_image,cv2.COLOR_YUV2BGR_Y422)
        original_image = cv2.flip(original_image, 0)
        image_data = cv2.resize(original_image, (input_size, input_size))
        b,g,r = cv2.split(image_data)
        image_data = np.stack([r,g,b]).astype(np.uint8)  

    elif '.npz' in flags.image:
        img_npz = np.load(str(image_path),allow_pickle=True)
        image_data = img_npz['img']
    elif '.npy' in flags.image:
        image_data = np.load(str(image_path),allow_pickle=True)
        print(image_data.shape)
    else:
        print(image_path)
        original_image = cv2.imread(str(image_path))
        image_data = cv2.resize(original_image, (input_size, input_size))
        b,g,r = cv2.split(image_data)
        image_data = np.stack([r,g,b]).astype(np.uint8)

    Net = nn.Sequential(command_path)
    buffer = Net.input(3,input_size,input_size)
    #buffer = Net.input(128,24,24)

    #%% backbone 
    buffer = Net.conv2d(buffer,16,(3,3),max_pooling=True,command_file='conv2d_0.npz') #208
    buffer = Net.conv2d(buffer,32,(3,3),max_pooling=True,command_file='conv2d_1.npz') #104
    buffer = Net.conv2d(buffer,64,(3,3),max_pooling=True,command_file='conv2d_2.npz') #52
    buffer = Net.conv2d(buffer,128,(3,3),max_pooling=True,command_file='conv2d_3.npz') #26
    buffer = Net.conv2d(buffer,256,(3,3),command_file='conv2d_4.npz') #26
    route_1 = buffer
    buffer = Net.maxpool2d(buffer, strides=(2,2)) # Software pooling (CPU)
    buffer = Net.conv2d(buffer,512,(3,3),command_file='conv2d_5.npz') #13
    buffer = Net.maxpool2d(buffer, strides=(1,1)) # Software pooling (CPU)
    buffer = Net.conv2d(buffer,1024,(3,3),command_file='conv2d_6.npz') #13

    # # #%% head 
    buffer = Net.conv2d(buffer,256,(1,1),command_file='conv2d_7.npz') #13

    lobj = buffer
    conv_lobj_branch = Net.conv2d(lobj,512,(3,3),command_file='conv2d_8.npz') #13
    conv_lbbox = Net.conv2d(conv_lobj_branch,256,(1,1),command_file='conv2d_9.npz') #13

    buffer = Net.conv2d(buffer,128,(1,1),command_file='conv2d_10.npz') #13
    buffer = Net.upsample(buffer) 
    buffer = Net.concat(buffer,route_1) #26
    # buffer = Net.copy(buffer)

    mobj = buffer
    conv_mobj_branch = Net.conv2d(mobj,256,(3,3),command_file='conv2d_11.npz') #26
    conv_mbbox = Net.conv2d(conv_mobj_branch,256,(1,1),command_file='conv2d_12.npz') #26  

    Net.output(conv_lbbox)
    Net.output(conv_mbbox)  


    Net.summary()
    if print_last:
        Net.print_layer_dma_info(layer_nbr)
    if exectue_net:    
        for i in range(flags.iterations):
            res = Net(image_data)
        if out_path.exists():
            shutil.rmtree(out_path, ignore_errors=False, onerror=None)    
        out_path.mkdir(parents=True, exist_ok=True)    
        # outfile_name = 'fmap_out_' + str(layer_nbr) + '.npy'
        # np.save(str(out_path / outfile_name), res)
        outfile_name = 'out_lbbox.npy'
        np.save(str(out_path / outfile_name), res[0])
        outfile_name = 'out_mbbox.npy'
        np.save(str(out_path / outfile_name), res[1])

if __name__ == '__main__':
    flags = _create_parser()
    main(flags)
