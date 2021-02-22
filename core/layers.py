import tensorflow as tf
from tensorflow.python.keras.utils import tf_utils
from core.yolov4 import filter_boxes

class Filter_boxes(tf.keras.layers.Layer):
    """A Keras layer that applies Filter_boxes.
    """

    def __init__(self,
                 filter_input_shape = tf.constant([416,416]), 
                 score_threshold=float('-inf'),
                 swap_tensors=True,
                 **kwargs):
        
        super(Filter_boxes,self).__init__(**kwargs)       

        self.filter_input_shape = filter_input_shape
        self.score_threshold = score_threshold
        self.swap_tensors = swap_tensors


    def call(self,pred):
        
        if self.swap_tensors == True:
            box_xywh = pred[1]
            scores = pred[0]
        else:
            box_xywh = pred[0]
            scores = pred[1]
        
        scores_max = tf.math.reduce_max(scores, axis=-1)
    
        mask = (scores_max >= self.score_threshold)
        class_boxes = tf.boolean_mask(box_xywh, mask)
        pred_conf = tf.boolean_mask(scores, mask)
        class_boxes = tf.reshape(class_boxes, [tf.shape(scores)[0], -1, tf.shape(class_boxes)[-1]])
        #pred_conf = tf.reshape(pred_conf, [tf.shape(scores)[0], -1, tf.shape(pred_conf)[-1]])
    
        box_xy, box_wh = tf.split(class_boxes, (2, 2), axis=-1)
    
        input_shape = tf.cast(self.filter_input_shape, dtype=tf.float32)
    
        box_yx = box_xy[..., ::-1]
        box_hw = box_wh[..., ::-1]
    
        box_mins = (box_yx - (box_hw / 2.)) / input_shape
        box_maxes = (box_yx + (box_hw / 2.)) / input_shape
        boxes = tf.concat([
            box_mins[..., 0:1],  # y_min
            box_mins[..., 1:2],  # x_min
            box_maxes[..., 0:1],  # y_max
            box_maxes[..., 1:2]  # x_max
        ], axis=-1)
        # return tf.concat([boxes, pred_conf], axis=-1)
        return (boxes, pred_conf)

class Combined_non_max_suppression(tf.keras.layers.Layer):
    """A Keras layer that applies Combined_non_max_suppression.
    """

    def __init__(self,
                 max_output_size,
                 iou_threshold=0.5,
                 score_threshold=float('-inf'),
                 soft_nms_sigma=0.0, 
                 **kwargs):
        
        super(Combined_non_max_suppression,self).__init__(**kwargs)       

        self.max_output_size = max_output_size
        self.iou_threshold = iou_threshold
        self.score_threshold = score_threshold
        self.soft_nms_sigma = soft_nms_sigma
    
    # @tf_utils.shape_type_conversion
    # def compute_output_shape(self, input_shape):
    #     if ((not isinstance(input_shape, (tuple, list))) or
    #         (not isinstance(input_shape[0], (tuple, list)))):
    #       # The tf_utils.shape_type_conversion decorator turns tensorshapes
    #       # into tuples, so we need to verify that `input_shape` is a list/tuple,
    #       # *and* that the individual elements are themselves shape tuples.
    #       raise ValueError('A `Concatenate` layer should be called '
    #                        'on a list of inputs.')
          
    #     nmsed_boxes_shape = list(input_shape).append(4)
    #     output_shape = [nmsed_boxes_shape, input_shape, input_shape, input_shape[0]]
    #     return tuple(output_shape)

    # def get_config(self):

    #     config = super().get_config().copy()
    #     config.update({
    #         'max_output_size_per_class': self.max_output_size_per_class,
    #         'max_total_size': self.max_total_size,
    #         'filter_input_shape': self.filter_input_shape,
    #         'iou_threshold': self.iou_threshold,
    #         'swap_pred': self.swap_pred,
    #         'score_threshold': self.score_threshold,
    #         'pad_per_class': self.pad_per_class,
    #         'clip_boxes': self.clip_boxes,
    #     })
    #     return config

    def call(self,boxes, pred_conf):
        pd_re = tf.reshape(pred_conf, (tf.shape(pred_conf)[0]))
        print(tf.shape(pd_re))
        
        return tf.image.non_max_suppression_with_scores(
                    boxes= tf.reshape(boxes,(tf.shape(boxes)[0], 4)),
                    scores=pd_re,
                    max_output_size = self.max_output_size,
                    iou_threshold=self.iou_threshold,
                    score_threshold=self.score_threshold,
                    soft_nms_sigma=self.soft_nms_sigma)


def convert_to_corners(boxes):
    """Changes the box format to corner coordinates

    Arguments:
      boxes: A tensor of rank 2 or higher with a shape of `(..., num_boxes, 4)`
        representing bounding boxes where each box is of the format
        `[x, y, width, height]`.

    Returns:
      converted boxes with shape same as that of boxes.
    """
    return tf.concat(
        [boxes[..., :2] - boxes[..., 2:] / 2.0, boxes[..., :2] + boxes[..., 2:] / 2.0],
        axis=-1,
    )

class AnchorBox:
    """Generates anchor boxes.

    This class has operations to generate anchor boxes for feature maps at
    strides `[8, 16, 32, 64, 128]`. Where each anchor each box is of the
    format `[x, y, width, height]`.

    Attributes:
      aspect_ratios: A list of float values representing the aspect ratios of
        the anchor boxes at each location on the feature map
      scales: A list of float values representing the scale of the anchor boxes
        at each location on the feature map.
      num_anchors: The number of anchor boxes at each location on feature map
      areas: A list of float values representing the areas of the anchor
        boxes for each feature map in the feature pyramid.
      strides: A list of float value representing the strides for each feature
        map in the feature pyramid.
    """

    def __init__(self):
        self.aspect_ratios = [0.5, 1.0, 2.0]
        self.scales = [2 ** x for x in [0, 1 / 3, 2 / 3]]

        self._num_anchors = len(self.aspect_ratios) * len(self.scales)
        self._strides = [2 ** i for i in range(3, 8)]
        self._areas = [x ** 2 for x in [32.0, 64.0, 128.0, 256.0, 512.0]]
        self._anchor_dims = self._compute_dims()

    def _compute_dims(self):
        """Computes anchor box dimensions for all ratios and scales at all levels
        of the feature pyramid.
        """
        anchor_dims_all = []
        for area in self._areas:
            anchor_dims = []
            for ratio in self.aspect_ratios:
                anchor_height = tf.math.sqrt(area / ratio)
                anchor_width = area / anchor_height
                dims = tf.reshape(
                    tf.stack([anchor_width, anchor_height], axis=-1), [1, 1, 2]
                )
                for scale in self.scales:
                    anchor_dims.append(scale * dims)
            anchor_dims_all.append(tf.stack(anchor_dims, axis=-2))
        return anchor_dims_all

    def _get_anchors(self, feature_height, feature_width, level):
        """Generates anchor boxes for a given feature map size and level

        Arguments:
          feature_height: An integer representing the height of the feature map.
          feature_width: An integer representing the width of the feature map.
          level: An integer representing the level of the feature map in the
            feature pyramid.

        Returns:
          anchor boxes with the shape
          `(feature_height * feature_width * num_anchors, 4)`
        """
        rx = tf.range(feature_width, dtype=tf.float32) + 0.5
        ry = tf.range(feature_height, dtype=tf.float32) + 0.5
        centers = tf.stack(tf.meshgrid(rx, ry), axis=-1) * self._strides[level - 3]
        centers = tf.expand_dims(centers, axis=-2)
        centers = tf.tile(centers, [1, 1, self._num_anchors, 1])
        dims = tf.tile(
            self._anchor_dims[level - 3], [feature_height, feature_width, 1, 1]
        )
        anchors = tf.concat([centers, dims], axis=-1)
        return tf.reshape(
            anchors, [feature_height * feature_width * self._num_anchors, 4]
        )

    def get_anchors(self, image_height, image_width):
        """Generates anchor boxes for all the feature maps of the feature pyramid.

        Arguments:
          image_height: Height of the input image.
          image_width: Width of the input image.

        Returns:
          anchor boxes for all the feature maps, stacked as a single tensor
            with shape `(total_anchors, 4)`
        """
        anchors = [
            self._get_anchors(
                tf.math.ceil(image_height / 2 ** i),
                tf.math.ceil(image_width / 2 ** i),
                i,
            )
            for i in range(3, 8)
        ]
        return tf.concat(anchors, axis=0)
    
class DecodePredictions(tf.keras.layers.Layer):
    """A Keras layer that decodes predictions of the RetinaNet model.

    Attributes:
      num_classes: Number of classes in the dataset
      confidence_threshold: Minimum class probability, below which detections
        are pruned.
      nms_iou_threshold: IOU threshold for the NMS operation
      max_detections_per_class: Maximum number of detections to retain per
       class.
      max_detections: Maximum number of detections to retain across all
        classes.
      box_variance: The scaling factors used to scale the bounding box
        predictions.
    """

    def __init__(
        self,
        num_classes=80,
        confidence_threshold=0.05,
        nms_iou_threshold=0.5,
        max_detections_per_class=100,
        max_detections=100,
        box_variance=[0.1, 0.1, 0.2, 0.2],
        **kwargs
    ):
        super(DecodePredictions, self).__init__(**kwargs)
        self.num_classes = num_classes
        self.confidence_threshold = confidence_threshold
        self.nms_iou_threshold = nms_iou_threshold
        self.max_detections_per_class = max_detections_per_class
        self.max_detections = max_detections

        self._anchor_box = AnchorBox()
        self._box_variance = tf.convert_to_tensor(
            [0.1, 0.1, 0.2, 0.2], dtype=tf.float32
        )

    def _decode_box_predictions(self, anchor_boxes, box_predictions):
        boxes = box_predictions * self._box_variance
        boxes = tf.concat(
            [
                boxes[:, :, :2] * anchor_boxes[:, :, 2:] + anchor_boxes[:, :, :2],
                tf.math.exp(boxes[:, :, 2:]) * anchor_boxes[:, :, 2:],
            ],
            axis=-1,
        )
        boxes_transformed = convert_to_corners(boxes)
        return boxes_transformed

    def call(self, images, predictions):
        image_shape = tf.cast(tf.shape(images), dtype=tf.float32)
        anchor_boxes = self._anchor_box.get_anchors(image_shape[1], image_shape[2])
        box_predictions = predictions[0]
        cls_predictions = tf.nn.sigmoid(predictions[1])
        boxes = self._decode_box_predictions(anchor_boxes[None, ...], box_predictions)

        return tf.image.combined_non_max_suppression(
            tf.expand_dims(boxes, axis=2),
            cls_predictions,
            self.max_detections_per_class,
            self.max_detections,
            self.nms_iou_threshold,
            self.confidence_threshold,
            clip_boxes=False,
        )