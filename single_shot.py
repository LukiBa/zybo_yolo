import argparse
import core.utils as utils
import cv2
import numpy as np

try: import tflite_runtime.interpreter as tf_itp
except: import tensorflow.lite as tf_itp

try: from intuitus_intf import Framebuffer, Camera
except: pass

def _create_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, default='./checkpoints/yolov4-tiny-416.tflite', help='path to weights file')
    parser.add_argument('--image',  type=str, default='./cam_data/dog.jpg', help='path to input image')
    parser.add_argument('--output',  type=str, default='./result.png', help='path to output')
    parser.add_argument('--tiny', type=bool, default=True, help='is yolo-tiny or not')
    parser.add_argument('--size', type=int, default=416, help='define input size of export model')
    parser.add_argument('--iou', type=float, default=0.45, help='iou threshold')
    parser.add_argument('--score', type=float, default=0.25, help='score threshold')
    parser.add_argument('--use_cam', type=bool, default=False, help='use camera for input img')
    parser.add_argument('--use_fb', type=bool, default=False, help='write output to framebuffer')
    parser.add_argument('--model', type=str, default='yolov4', help='yolov3 or yolov4')
    return parser.parse_args()   

def main(flags):

    input_size = flags.size
    image_path = flags.image

    if flags.use_cam: 
        cam = Camera('/dev/video0')
        status, original_image = cam.capture()
        print(status)
        print(original_image.shape)
        original_image = cv2.cvtColor(original_image,cv2.COLOR_YUV2RGB_Y422)
        original_image = cv2.flip(original_image, 0)
    else:
        original_image = cv2.imread(image_path)
        original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)

    # image_data = utils.image_preprocess(np.copy(original_image), [input_size, input_size])
    image_data = cv2.resize(original_image, (input_size, input_size))
    image_data = image_data / 255.
    # image_data = image_data[np.newaxis, ...].astype(np.float32)

    images_data = []
    for i in range(1):
        images_data.append(image_data)
    images_data = np.asarray(images_data).astype(np.float32)

    interpreter = tf_itp.Interpreter(model_path=flags.model_path)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    print(input_details)
    print(output_details)
    interpreter.set_tensor(input_details[0]['index'], images_data)
    interpreter.invoke()
    pred = [interpreter.get_tensor(output_details[i]['index']) for i in range(len(output_details))]
    if flags.model == 'yolov3' and flags.tiny == True:
        boxes, pred_conf = utils.filter_boxes_np(pred[1], pred[0], score_threshold=flags.score, input_shape=[input_size, input_size])
    else:
        boxes, pred_conf = utils.filter_boxes_np(pred[0], pred[1], score_threshold=flags.score, input_shape=[input_size, input_size])
    
    np_boxes = boxes[0,:,:]
    scores = np.max(pred_conf[0,:,:],axis=-1)
    classes = np.argmax(pred_conf[0,:,:],axis=-1)
    bboxes = np.concatenate([np_boxes,scores.reshape([scores.shape[0],1]),
                             classes.reshape([scores.shape[0],1])],axis=1)

    nms_boxes = utils.nms(bboxes,flags.iou)
    out_boxes = [i[:4] for i in nms_boxes]
    out_socres = [i[4] for i in nms_boxes]
    out_classes = [i[5] for i in nms_boxes]
    pred_bbox = [out_boxes, out_socres, out_classes]
    image = utils.draw_bbox_nms(original_image, pred_bbox)
    
    if flags.use_fb:
        fb = Framebuffer('/dev/fb0')
        black_screen = np.zeros([1080,1920,3],dtype=np.uint8)
        fb.show(black_screen,0)
        print(image.shape)  
        img_bgr = cv2.cvtColor(image,cv2.COLOR_RGB2BGR)
        fb.show(img_bgr,0) # (1920*20+500)*3

    image = cv2.cvtColor(np.array(image), cv2.COLOR_BGR2RGB)
    cv2.imwrite(flags.output, image)

if __name__ == '__main__':
    flags = _create_parser()
    main(flags)
