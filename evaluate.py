import argparse
import cv2
import os
import shutil
import numpy as np
import tensorflow as tf
from core.yolov4 import filter_boxes
from tensorflow.python.saved_model import tag_constants
import core.utils as utils
from core.config import cfg
import pathlib
from core.dataset import Dataset

def _create_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, default='./checkpoints/yolov4-tiny-416.tflite', help='path to saved model')
    parser.add_argument('--framework', type=str, default='tflite', help='define what framework do you want to convert (tf, trt, tflite)')
    parser.add_argument('--model', type=str, default='yolov4', help='yolov3 or yolov4')
    parser.add_argument('--tiny', type=bool, default=True, help='is yolo-tiny or not')
    parser.add_argument('--size', type=int, default=416, help='define input size of export model')
    parser.add_argument('--annotation_path',  type=str, default='../../Datasets/coco/val2017.txt', help='path to output')
    parser.add_argument('--write_image_path',  type=str, default='./data/detection/', help='write image path')
    parser.add_argument('--iou', type=float, default=0.45, help='iou threshold')
    parser.add_argument('--score', type=float, default=0.25, help='score threshold')
    parser.add_argument('--max_img', type=int, default=100, help='maximum image to be processed. -1 for infinit.')
    parser.add_argument('--io_batch_size', type=int, default=10, help='number of images which are loaded in parallel')
    return parser.parse_args()

def main(flags):
    INPUT_SIZE = flags.size
    STRIDES, ANCHORS, NUM_CLASS, XYSCALE = utils.load_config(flags.model, flags.tiny)
    CLASSES = utils.read_class_names(cfg.YOLO.CLASSES)

    mAP_path = pathlib.Path(__file__).parent / 'mAP'
    mAP_path.mkdir(parents=True, exist_ok=True)
    ground_truth_dir_path = (mAP_path / 'ground-truth').absolute()
    predicted_dir_path =( mAP_path / 'predicted').absolute()
    detect_img_path = pathlib.Path(cfg.TEST.DECTECTED_IMAGE_PATH).absolute()
    test_annot_path = pathlib.Path(cfg.TEST.ANNOT_PATH).absolute()
    if not test_annot_path.exists():
        raise FileNotFoundError(str(test_annot_path) + " does not exist. Check config.py.")
    if os.path.exists(predicted_dir_path): 
        shutil.rmtree(predicted_dir_path)
    if os.path.exists(ground_truth_dir_path): 
        shutil.rmtree(ground_truth_dir_path)
    if os.path.exists(detect_img_path): 
        shutil.rmtree(detect_img_path)
  
    predicted_dir_path.mkdir(parents=True, exist_ok=True)  
    ground_truth_dir_path.mkdir(parents=True, exist_ok=True)
    detect_img_path.mkdir(parents=True, exist_ok=True)
    pathlib.Path('/tmp/sub1/sub2').mkdir(parents=True, exist_ok=True)
    
    testset = Dataset(flags, is_training=False)

    
    print("Detect predicted_dir_path: " + str(predicted_dir_path))
    print("Detect ground_truth_dir_path: " + str(ground_truth_dir_path))
    print("Detect image path: " + str(detect_img_path))
    

    # Build Model
    if flags.framework == 'tflite':
        interpreter = tf.lite.Interpreter(model_path=flags.model_path)
        interpreter.allocate_tensors()
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
        print(input_details)
        print(output_details)
    else:
        saved_model_loaded = tf.saved_model.load(flags.model_path, tags=[tag_constants.SERVING])
        infer = saved_model_loaded.signatures['serving_default']

    num_lines = sum(1 for line in open(flags.annotation_path))
    if flags.io_batch_size > 0 and flags.io_batch_size < num_lines:
        num_lines = flags.io_batch_size           
                   
        
    print(test_annot_path)
    with open(test_annot_path, 'r') as annotation_file:
        for num, line in enumerate(annotation_file):
            annotation = line.strip().split()
            image_path = annotation[0]
            image_path = pathlib.Path(image_path).absolute()
            image_name = image_path.name #image_path.split('/')[-1]
            image = cv2.imread(str(image_path))
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            bbox_data_gt = np.array([list(map(int, box.split(','))) for box in annotation[1:]])

            if len(bbox_data_gt) == 0:
                bboxes_gt = []
                classes_gt = []
            else:
                bboxes_gt, classes_gt = bbox_data_gt[:, :4], bbox_data_gt[:, 4]
            ground_truth_path = os.path.join(ground_truth_dir_path, str(num) + '.txt')

            print('=> ground truth of %s:' % image_name)
            num_bbox_gt = len(bboxes_gt)
            with open(ground_truth_path, 'w') as f:
                for i in range(num_bbox_gt):
                    class_name = CLASSES[classes_gt[i]]
                    xmin, ymin, xmax, ymax = list(map(str, bboxes_gt[i]))
                    bbox_mess = ' '.join([class_name, xmin, ymin, xmax, ymax]) + '\n'
                    f.write(bbox_mess)
                    print('\t' + str(bbox_mess).strip())
            print('=> predict result of %s:' % image_name)
            predict_result_path = os.path.join(predicted_dir_path, str(num) + '.txt')
            # Predict Process
            image_size = image.shape[:2]
            # image_data = utils.image_preprocess(np.copy(image), [INPUT_SIZE, INPUT_SIZE])
            image_data = cv2.resize(np.copy(image), (INPUT_SIZE, INPUT_SIZE))
            image_data = image_data / 255.
            image_data = image_data[np.newaxis, ...].astype(np.float32)

            if flags.framework == 'tflite':
                interpreter.set_tensor(input_details[0]['index'], image_data)
                interpreter.invoke()
                pred = [interpreter.get_tensor(output_details[i]['index']) for i in range(len(output_details))]
                if flags.model == 'yolov3' and flags.tiny == True:
                    boxes, pred_conf = filter_boxes(pred[1], pred[0], score_threshold=flags.score, input_shape=tf.constant([INPUT_SIZE,INPUT_SIZE]))
                else:
                    boxes, pred_conf = filter_boxes(pred[0], pred[1], score_threshold=flags.score, input_shape=tf.constant([INPUT_SIZE,INPUT_SIZE]))    
            else:
                batch_data = tf.constant(image_data)
                pred_bbox = infer(batch_data)
                for key, value in pred_bbox.items():
                    boxes = value[:, :, 0:4]
                    pred_conf = value[:, :, 4:]                

            boxes, scores, classes, valid_detections = tf.image.combined_non_max_suppression(
                boxes=tf.reshape(boxes, (tf.shape(boxes)[0], -1, 1, 4)),
                scores=tf.reshape(
                    pred_conf, (tf.shape(pred_conf)[0], -1, tf.shape(pred_conf)[-1])),
                max_output_size_per_class=50,
                max_total_size=50,
                iou_threshold=flags.iou,
                score_threshold=flags.score
            )
            boxes, scores, classes, valid_detections = [boxes.numpy(), scores.numpy(), classes.numpy(), valid_detections.numpy()]

            # if detect_img_path is not None:
            #     image_result = utils.draw_bbox(np.copy(image), [boxes, scores, classes, valid_detections])
            #     cv2.imwrite(detect_img_path + image_name, image_result)

            with open(predict_result_path, 'w') as f:
                image_h, image_w, _ = image.shape
                for i in range(valid_detections[0]):
                    if int(classes[0][i]) < 0 or int(classes[0][i]) > NUM_CLASS: continue
                    coor = boxes[0][i]
                    coor[0] = int(coor[0] * image_h)
                    coor[2] = int(coor[2] * image_h)
                    coor[1] = int(coor[1] * image_w)
                    coor[3] = int(coor[3] * image_w)

                    score = scores[0][i]
                    class_ind = int(classes[0][i])
                    class_name = CLASSES[class_ind]
                    score = '%.4f' % score
                    ymin, xmin, ymax, xmax = list(map(str, coor))
                    bbox_mess = ' '.join([class_name, score, xmin, ymin, xmax, ymax]) + '\n'
                    f.write(bbox_mess)
                    print('\t' + str(bbox_mess).strip())
            print(num, num_lines)
            if num > num_lines:
                break


    # for num, image_data, targets in testset:
        
    #     for i in range(image_data.shape[0]):
    #         image = image_data[i:i+1,...]
    #         target =  targets
    #         if flags.framework == 'tflite':
    #             interpreter.set_tensor(input_details[0]['index'], image)
    #             interpreter.invoke()
    #             pred = [interpreter.get_tensor(output_details[i]['index']) for i in range(len(output_details))]
    #             if flags.model == 'yolov3' and flags.tiny == True:
    #                 boxes, pred_conf = filter_boxes(pred[1], pred[0], score_threshold=flags.score, input_shape=tf.constant([INPUT_SIZE,INPUT_SIZE]))
    #             else:
    #                 boxes, pred_conf = filter_boxes(pred[0], pred[1], score_threshold=flags.score, input_shape=tf.constant([INPUT_SIZE,INPUT_SIZE]))    
    #         else:
    #             batch_data = tf.constant(image)
    #             pred_bbox = infer(batch_data)
    #             for key, value in pred_bbox.items():
    #                 boxes = value[:, :, 0:4]
    #                 pred_conf = value[:, :, 4:]                
    
    #         boxes, scores, classes, valid_detections = tf.image.combined_non_max_suppression(
    #             boxes=tf.reshape(boxes, (tf.shape(boxes)[0], -1, 1, 4)),
    #             scores=tf.reshape(
    #                 pred_conf, (tf.shape(pred_conf)[0], -1, tf.shape(pred_conf)[-1])),
    #             max_output_size_per_class=50,
    #             max_total_size=50,
    #             iou_threshold=flags.iou,
    #             score_threshold=flags.score
    #         )
    #         boxes, scores, classes, valid_detections = [boxes.numpy(), scores.numpy(), classes.numpy(), valid_detections.numpy()]
    
    #         giou_loss = conf_loss = prob_loss = 0
    #         # for i in range(len(2)):
    #         #     conv, pred = pred_result[i * 2], pred_result[i * 2 + 1]
    #         #     loss_items = compute_loss(pred, conv, target[i][0], target[i][1], STRIDES=STRIDES, NUM_CLASS=NUM_CLASS, IOU_LOSS_THRESH=IOU_LOSS_THRESH, i=i)
    #         #     giou_loss += loss_items[0]
    #         #     conf_loss += loss_items[1]
    #         #     prob_loss += loss_items[2]
    
    #         # total_loss = giou_loss + conf_loss + prob_loss
    
    #         # tf.print("=> TEST STEP %4d   giou_loss: %4.2f   conf_loss: %4.2f   "
    #         #          "prob_loss: %4.2f   total_loss: %4.2f" % (global_steps, giou_loss, conf_loss,
    #         #                                                    prob_loss, total_loss))
            
    #         predict_result_path = os.path.join(predicted_dir_path, str(num) + '.txt')
    #         with open(predict_result_path, 'w') as f:
    #             image_h, image_w, _ = image.shape
    #             for i in range(valid_detections[0]):
    #                 if int(classes[0][i]) < 0 or int(classes[0][i]) > NUM_CLASS: continue
    #                 coor = boxes[0][i]
    #                 coor[0] = int(coor[0] * image_h)
    #                 coor[2] = int(coor[2] * image_h)
    #                 coor[1] = int(coor[1] * image_w)
    #                 coor[3] = int(coor[3] * image_w)
    
    #                 score = scores[0][i]
    #                 class_ind = int(classes[0][i])
    #                 class_name = CLASSES[class_ind]
    #                 score = '%.4f' % score
    #                 ymin, xmin, ymax, xmax = list(map(str, coor))
    #                 bbox_mess = ' '.join([class_name, score, xmin, ymin, xmax, ymax]) + '\n'
    #                 f.write(bbox_mess)
    #                 print('\t' + str(bbox_mess).strip())
    #         print(num, num_lines)
    #         if num > num_lines:
    #             break

if __name__ == '__main__':
    flags = _create_parser()
    main(flags)


