import tensorflow as tf
import argparse
import numpy as np
import cv2
from core.yolov4 import YOLOv4, YOLOv3, YOLOv3_tiny, decode
import core.utils as utils
import os
from core.config import cfg

def _create_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', type=str, default='./checkpoints/yolov4-tiny-416', help='path to weights file')
    parser.add_argument('--output',  type=str, default='./checkpoints/yolov4-tiny-416.tflite', help='path to output')
    parser.add_argument('--input_size', type=int, default=416, help='define input size of export model')
    parser.add_argument('--quantize_mode', type=str, default='float32', help='quantize mode (int8, float16, float32)')
    parser.add_argument('--dataset', type=str, default="/Volumes/Elements/data/coco_dataset/coco/5k.txt", help='path to dataset')
    return parser.parse_args()

# def representative_data_gen():
#   fimage = open(flags.dataset).read().split()
#   for input_value in range(10):
#     if os.path.exists(fimage[input_value]):
#       original_image=cv2.imread(fimage[input_value])
#       original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
#       image_data = utils.image_preprocess(np.copy(original_image), [flags.input_size, flags.input_size])
#       img_in = image_data[np.newaxis, ...].astype(np.float32)
#       print("calibration image {}".format(fimage[input_value]))
#       yield [img_in]
#     else:
#       continue
def representative_data_gen(flags):
    fimage = open(flags.dataset).read().splitlines()
    images_list = []
    for i in range(100):
        img_path=fimage[i].split()
        original_image=cv2.imread(img_path[0])
        original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
        image_data = cv2.resize(original_image, (flags.input_size, flags.input_size))
        image_data = image_data / 255.
        images_list.append(image_data)
        
    images_list = np.asarray(images_list).astype(np.float32)    
    yield [images_list]


def save_tflite(flags):
  converter = tf.lite.TFLiteConverter.from_saved_model(flags.weights)

  if flags.quantize_mode == 'float16':
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.target_spec.supported_types = [tf.compat.v1.lite.constants.FLOAT16]
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS, tf.lite.OpsSet.SELECT_TF_OPS]
    converter.allow_custom_ops = True
  elif flags.quantize_mode == 'int8':
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS, tf.lite.OpsSet.SELECT_TF_OPS]
    converter.allow_custom_ops = True
    converter.representative_dataset = representative_data_gen

  tflite_model = converter.convert()
  open(flags.output, 'wb').write(tflite_model)

  print("model saved to: {}".format(flags.output))

def demo(flags):
  interpreter = tf.lite.Interpreter(model_path=flags.output)
  interpreter.allocate_tensors()
  print('tflite model loaded')

  input_details = interpreter.get_input_details()
  print(input_details)
  output_details = interpreter.get_output_details()
  print(output_details)

  input_shape = input_details[0]['shape']

  input_data = np.array(np.random.random_sample(input_shape), dtype=np.float32)

  interpreter.set_tensor(input_details[0]['index'], input_data)
  interpreter.invoke()
  output_data = [interpreter.get_tensor(output_details[i]['index']) for i in range(len(output_details))]

  print(output_data)

def main(flags):
  save_tflite(flags)
  demo(flags)

if __name__ == '__main__':
    flags = _create_parser()
    main(flags)


