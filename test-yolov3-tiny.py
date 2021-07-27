import numpy as np
import cv2
import pathlib
import argparse
import shutil
import timeit
from elevate import elevate
import os

from intuitus_nn import Framebuffer, Camera
import intuitus_nn as nn
from core.utils import YoloLayer, filter_boxes, draw_bbox_nms, nms, read_class_names



def _create_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--command_path',  type=str, default='./intuitus_commands/yolov3-tiny-commands', help='path to intuitus command files')
    parser.add_argument('--classes_path',  type=str, default='./data/classes/coco.names', help='path to class name file')
    parser.add_argument('--image',  type=str, default='./cam_data/000000000069.jpg', help='path to input image') # './cam_data/fmap384_out_4.npy') #
    parser.add_argument('--output',  type=str, default='./fmap_out', help='path to output')
    parser.add_argument('--tiny', type=bool, default=True, help='is yolo-tiny or not')
    parser.add_argument('--size', type=int, default=416, help='define input size of export model')
    parser.add_argument('--conf_thres', type=float, default=0.1, help='define confidence threshold')
    parser.add_argument('--iou_thres', type=float, default=0.4, help='define iou threshold')
    parser.add_argument('--score', type=float, default=0.15, help='object confidence threshold')
    parser.add_argument('--use_cam', action='store_true', help='use camera for input img')
    parser.add_argument('--use_fb', action='store_true', help='write output to framebuffer')
    parser.add_argument('--print_net', action='store_true', help='print network')
    parser.add_argument('--print_layer_info', type=int, default=None, help='print layer dma info. Expects layer number to be printed.')
    parser.add_argument('--save_result_boxes', action='store_true', help='save network output boxes as .npy file')
    parser.add_argument('--not_execute', action='store_true', help='print network')
    parser.add_argument('--iterations', type=int, default=1, help='execution iterations')
    
    return parser.parse_args() 


layer_nbr = 1

def _yolov3_tiny_config():
    yolo_lb = { 'anchors':np.array([[81,82],  [135,169],  [344,319]]),
                'stride': 32,
                'classes':80}

    yolo_mb = { 'anchors':np.array([[10,14],  [23,27],  [37,58]]),
                'stride': 16,
                'classes':80}

    yolo_conf = {   'lb':yolo_lb,
                    'mb':yolo_mb}
 
    return yolo_conf

def main(flags):
    if os.getuid() != 0:
        try: elevate() # get root privileges. Required to open kernel driver 
        except: raise Exception("Root permission denied") 
    exectue_net = not flags.not_execute 
    input_size = flags.size
    image_path = pathlib.Path(__file__).absolute().parent / flags.image
    command_path = pathlib.Path(__file__).absolute().parent / flags.command_path
    class_name_path = pathlib.Path(__file__).absolute().parent / flags.classes_path
    out_path = pathlib.Path(__file__).absolute().parent / flags.output
    if out_path.exists():
        shutil.rmtree(out_path, ignore_errors=False, onerror=None)    
    out_path.mkdir(parents=True, exist_ok=True)  

    if flags.tiny:
        yolo_config = _yolov3_tiny_config()
    else:
        raise NotImplementedError("Add config for non tiny implementaion")

    result_scale = 2.0**-4.0

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

    Yolo_lb = YoloLayer(yolo_config['lb']['anchors'],
                        yolo_config['lb']['classes'],
                        yolo_config['lb']['stride'])

    Yolo_mb = YoloLayer(yolo_config['mb']['anchors'],
                        yolo_config['mb']['classes'],
                        yolo_config['mb']['stride'])
    if flags.print_net:
        Net.summary()
    if flags.print_layer_info != None: 
        Net.print_layer_dma_info(flags.print_layer_info)

    if flags.use_cam:
        cam = Camera('/dev/video0')
    if flags.use_fb:
        fb = Framebuffer('/dev/fb0')        

    for i in range(flags.iterations):
        start = timeit.default_timer() 
        if flags.use_cam:
            img_start = timeit.default_timer() 
            status, original_image = cam.capture()
            img_cap = timeit.default_timer() 
            original_image = cv2.cvtColor(original_image,cv2.COLOR_YUV2BGR_Y422)
            original_image = cv2.flip(original_image, 0)
            original_image[...,1] = np.uint8(original_image[...,1]*0.66) # fix green channel error
            image_data = cv2.resize(original_image, (input_size, input_size))
            b,g,r = cv2.split(image_data)
            image_data = np.stack([r,g,b]).astype(np.uint8)
            img_time = timeit.default_timer() 
            print(  "Img capture time: {}ms.".format((img_cap-img_start)*1000) + \
                    "Img resize time: {}ms".format((img_time-img_cap)*1000))  

        elif '.npz' in flags.image:
            img_npz = np.load(str(image_path),allow_pickle=True)
            image_data = img_npz['img'].astype(np.uint8)
        elif '.npy' in flags.image:
            image_data = np.load(str(image_path),allow_pickle=True)
        else:
            original_image = cv2.imread(str(image_path))
            image_data = cv2.resize(original_image, (input_size, input_size))
            b,g,r = cv2.split(image_data)
            image_data = np.stack([r,g,b]).astype(np.uint8)

        
        
        if exectue_net:   
            nn_start = timeit.default_timer() 
            fmap_out = Net(image_data)
            cnn_time = timeit.default_timer()

            pred_lb = Yolo_lb(fmap_out[0][:255,...]*result_scale)
            pred_mb = Yolo_mb(fmap_out[1][:255,...]*result_scale)
            inf_out = np.concatenate((pred_lb,pred_mb),axis=0)
            yolo_layer_time = timeit.default_timer()

            boxes, pred_conf, classes = filter_boxes(inf_out,flags.conf_thres)
            best_bboxes = nms(boxes, pred_conf, classes, iou_threshold = flags.iou_thres, 
                            score=flags.score,method='merge')
            classes = read_class_names(str(class_name_path))
            image_data = np.moveaxis(image_data,0,-1).astype(np.uint8).copy('C')
            image = draw_bbox_nms(image_data, best_bboxes,classes)
            nms_time = timeit.default_timer()

            print(  "CNN Excution time: {}ms. \n".format((cnn_time-nn_start)*1000) + \
                    "YOLO layer(numpy): {}ms. \n".format((yolo_layer_time-cnn_time)*1000) + \
                    "NMS (numpy): {}ms. \n".format((nms_time-yolo_layer_time)*1000) + \
                    "Full time: {}ms".format((nms_time-nn_start)*1000))   
            if flags.save_result_boxes:                
                outfile_name = 'best_bboxes.npy'    
                np.save(str(out_path / outfile_name),best_bboxes)

        if flags.use_fb:
            plot_start= timeit.default_timer()
            img_bgr = cv2.cvtColor(image,cv2.COLOR_RGB2BGR)
            screen_size = fb.get_screensize()
            img_bgr = cv2.resize(img_bgr,(screen_size[1],screen_size[0]))
            fb.show(img_bgr,0)
            plot_time = timeit.default_timer()
            print(  "Plot time: {}ms. \n".format((plot_time-plot_start)*1000))
            
        else:
            image = cv2.cvtColor(np.array(image), cv2.COLOR_BGR2RGB)
            outfile_name = 'detect.png'  
            cv2.imwrite(str(out_path / outfile_name), image)            

if __name__ == '__main__':
    flags = _create_parser()
    main(flags)
