from absl import app, flags, logging
import core.utils as utils
from PIL import Image
import cv2
import numpy as np
from tflite_runtime.interpreter import Interpreter
from intuitus_intf import Framebuffer, Camera

# flags.DEFINE_string('weights', './checkpoints/yolov4-416',
#                     'path to weights file')
# flags.DEFINE_integer('size', 416, 'resize images to')
flags.DEFINE_boolean('tiny', False, 'yolo or yolo-tiny')
flags.DEFINE_string('model', 'yolov4', 'yolov3 or yolov4')
# flags.DEFINE_string('image', './cam_data/dog.jpg', 'path to input image')
# flags.DEFINE_string('output', 'result.png', 'path to output image')
# flags.DEFINE_float('iou', 0.45, 'iou threshold')
# flags.DEFINE_float('score', 0.25, 'score threshold')qui

model = 'yolov4'
tiny = 'True'
USE_CAM = False

def main(_argv):
    # config = ConfigProto()
    # config.gpu_options.allow_growth = True
    # session = InteractiveSession(config=config)
    STRIDES, ANCHORS, NUM_CLASS, XYSCALE = utils.load_config(flags)
    input_size = 416
    image_path = './cam_data/dog.jpg'

    if USE_CAM: 
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

    interpreter = Interpreter(model_path='./checkpoints/yolov4-416-tiny-fp16.tflite')
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    print(input_details)
    print(output_details)
    interpreter.set_tensor(input_details[0]['index'], images_data)
    interpreter.invoke()
    pred = [interpreter.get_tensor(output_details[i]['index']) for i in range(len(output_details))]
    if model == 'yolov3' and tiny == True:
        boxes, pred_conf = utils.filter_boxes_np(pred[1], pred[0], score_threshold=0.25, input_shape=[input_size, input_size])
    else:
        boxes, pred_conf = utils.filter_boxes_np(pred[0], pred[1], score_threshold=0.25, input_shape=[input_size, input_size])
    
    np_boxes = boxes[0,:,:]
    scores = np.max(pred_conf[0,:,:],axis=-1)
    classes = np.argmax(pred_conf[0,:,:],axis=-1)
    bboxes = np.concatenate([np_boxes,scores.reshape([scores.shape[0],1]),
                             classes.reshape([scores.shape[0],1])],axis=1)

    nms_boxes = utils.nms(bboxes,0.45)
    out_boxes = [i[:4] for i in nms_boxes]
    out_socres = [i[4] for i in nms_boxes]
    out_classes = [i[5] for i in nms_boxes]
    pred_bbox = [out_boxes, out_socres, out_classes]
    image = utils.draw_bbox_nms(original_image, pred_bbox)
    #image = Image.fromarray(image.astype(np.uint8))
    #image.show()
    
    fb = Framebuffer('/dev/fb0')
    black_screen = np.zeros([1080,1920,3],dtype=np.uint8)
    fb.update(black_screen,0)
    print(image.shape)  
    img_bgr = cv2.cvtColor(image,cv2.COLOR_RGB2BGR)
    fb.update(img_bgr,0) # (1920*20+500)*3

    image = cv2.cvtColor(np.array(image), cv2.COLOR_BGR2RGB)
    cv2.imwrite('result.png', image)

if __name__ == '__main__':
    try:
        app.run(main)
    except SystemExit:
        pass
