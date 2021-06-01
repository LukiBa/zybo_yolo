import cv2
import random
import colorsys
import numpy as np
from core.config import cfg
import pathlib
import json

def load_freeze_layer(model='yolov4', tiny=False):
    if tiny:
        if model == 'yolov3':
            freeze_layouts = ['conv2d_9', 'conv2d_12']
        else:
            freeze_layouts = ['conv2d_17', 'conv2d_20']
    else:
        if model == 'yolov3':
            freeze_layouts = ['conv2d_58', 'conv2d_66', 'conv2d_74']
        else:
            freeze_layouts = ['conv2d_93', 'conv2d_101', 'conv2d_109']
    return freeze_layouts

def load_weights(model, weights_file, model_name='yolov4', is_tiny=False):
    if is_tiny:
        if model_name == 'yolov3':
            layer_size = 13
            output_pos = [9, 12]
        else:
            layer_size = 21
            output_pos = [17, 20]
    else:
        if model_name == 'yolov3':
            layer_size = 75
            output_pos = [58, 66, 74]
        else:
            layer_size = 110
            output_pos = [93, 101, 109]
    wf = open(weights_file, 'rb')
    major, minor, revision, seen, _ = np.fromfile(wf, dtype=np.int32, count=5)

    j = 0
    for i in range(layer_size):
        conv_layer_name = 'conv2d_%d' %i if i > 0 else 'conv2d'
        bn_layer_name = 'batch_normalization_%d' %j if j > 0 else 'batch_normalization'

        conv_layer = model.get_layer(conv_layer_name)
        filters = conv_layer.filters
        k_size = conv_layer.kernel_size[0]
        in_dim = conv_layer.input_shape[-1]

        if i not in output_pos:
            # darknet weights: [beta, gamma, mean, variance]
            bn_weights = np.fromfile(wf, dtype=np.float32, count=4 * filters)
            # tf weights: [gamma, beta, mean, variance]
            bn_weights = bn_weights.reshape((4, filters))[[1, 0, 2, 3]]
            bn_layer = model.get_layer(bn_layer_name)
            j += 1
        else:
            conv_bias = np.fromfile(wf, dtype=np.float32, count=filters)

        # darknet shape (out_dim, in_dim, height, width)
        conv_shape = (filters, in_dim, k_size, k_size)
        conv_weights = np.fromfile(wf, dtype=np.float32, count=np.product(conv_shape))
        # tf shape (height, width, in_dim, out_dim)
        conv_weights = conv_weights.reshape(conv_shape).transpose([2, 3, 1, 0])

        if i not in output_pos:
            conv_layer.set_weights([conv_weights])
            bn_layer.set_weights(bn_weights)
        else:
            conv_layer.set_weights([conv_weights, conv_bias])

    # assert len(wf.read()) == 0, 'failed to read all data'
    wf.close()


def read_class_names(class_file_name):
    names = {}
    with open(class_file_name, 'r') as data:
        for ID, name in enumerate(data):
            names[ID] = name.strip('\n')
    return names

def load_config(model='yolov4',tiny=False):
    if tiny:
        STRIDES = np.array(cfg.YOLO.STRIDES_TINY)
        ANCHORS = get_anchors(cfg.YOLO.ANCHORS_TINY, tiny)
        XYSCALE = cfg.YOLO.XYSCALE_TINY if model == 'yolov4' else [1, 1]
    else:
        STRIDES = np.array(cfg.YOLO.STRIDES)
        if model == 'yolov4':
            ANCHORS = get_anchors(cfg.YOLO.ANCHORS, tiny)
        elif model == 'yolov3':
            ANCHORS = get_anchors(cfg.YOLO.ANCHORS_V3, tiny)
        else:
            Exception('Model not supported')
            ANCHORS = get_anchors(cfg.YOLO.ANCHORS_V3, tiny)
        XYSCALE = cfg.YOLO.XYSCALE if model == 'yolov4' else [1, 1, 1]
    NUM_CLASS = len(read_class_names(cfg.YOLO.CLASSES))

    return STRIDES, ANCHORS, NUM_CLASS, XYSCALE

def load_weights_folding_batchnorm(model, weights_file, model_name='yolov4', is_tiny=False):
    eps = 1e-3
    if is_tiny:
        if model_name == 'yolov3':
            layer_size = 13
            output_pos = [9, 12]
        else:
            layer_size = 21
            output_pos = [17, 20]
    else:
        if model_name == 'yolov3':
            layer_size = 75
            output_pos = [58, 66, 74]
        else:
            layer_size = 110
            output_pos = [93, 101, 109]
    wf = open(weights_file, 'rb')
    major, minor, revision, seen, _ = np.fromfile(wf, dtype=np.int32, count=5)

    j = 0
    for i in range(layer_size):
        conv_layer_name = 'conv2d_%d' %i #if i > 0 else 'conv2d'
        bn_layer_name = 'batch_normalization_%d' %j if j > 0 else 'batch_normalization'

        conv_layer = model.get_layer(conv_layer_name)
        
        filters = conv_layer.filters
        k_size = conv_layer.kernel_size[0]
        in_dim = conv_layer.input_shape[-1]

        if i not in output_pos:
            # darknet weights: [beta, gamma, mean, variance]
            bn_weights = np.fromfile(wf, dtype=np.float32, count=4 * filters)
            bn_weights = bn_weights.reshape((4, filters))
            [beta, gamma, mean, variance] = [bn_weights[0],bn_weights[1],bn_weights[2],bn_weights[3]]
            conv_bias = np.zeros(beta.shape) # only for completeness --> in fact useless 
            j += 1
        else:
            conv_bias = np.fromfile(wf, dtype=np.float32, count=filters)

        # darknet shape (out_dim, in_dim, height, width)
        conv_shape = (filters, in_dim, k_size, k_size)
        conv_weights = np.fromfile(wf, dtype=np.float32, count=np.product(conv_shape))
        # tf shape (height, width, in_dim, out_dim)
        conv_weights = conv_weights.reshape(conv_shape).transpose([2, 3, 1, 0])

        if i not in output_pos:
            
            fbn_bias = beta + (conv_bias-mean) * (gamma/np.sqrt(variance+eps))
            fbn_weights = conv_weights * (gamma/np.sqrt(variance+eps))
            
            conv_layer.set_weights([conv_weights, conv_bias])
        else:
            conv_layer.set_weights([conv_weights, conv_bias])

    # assert len(wf.read()) == 0, 'failed to read all data'
    wf.close()

def reshape_weights(weights,filters,k_size,in_channels,torch_weights_channels_first=True):
    if torch_weights_channels_first:
        assert weights.shape == (filters,in_channels,k_size,k_size), "Dimension missmatch reading weights with shape {} expected {}".format(weights.shape,(filters,in_channels,k_size,k_size))
        resh_weights = np.zeros((k_size,k_size,in_channels,filters))
        for i in range(in_channels):
            for j in range(filters):
                resh_weights[...,i,j] = weights[j,i,...] 
        return resh_weights
    else:
        assert weights.shape == (k_size,k_size,in_channels,filters), "Dimension missmatch reading weights with shape {} expected {}".format(weights.shape,(k_size,k_size,in_channels,filters))
        return weights
        
def load_weights_torch_npy_fb(model, weights_path, model_name='yolov4', is_tiny=False, keras2torch_path = './keras_2_torch_names.json',torch_weights_channels_first=True):
    weights_path = pathlib.Path(weights_path).absolute()
    keras2torch_path = pathlib.Path(keras2torch_path).absolute()
    with open(keras2torch_path) as json_file:
        data = json.load(json_file)
        if is_tiny:
            if model_name == 'yolov3':
                layer_size = 13
                tabular = data['yolov3-tiny']
            else:
                layer_size = 21
                tabular = data['yolov4-tiny']
        else:
            if model_name == 'yolov3':
                layer_size = 75
                tabular = data['yolov3']
            else:
                layer_size = 110
                tabular = data['yolov4']

    for i in range(layer_size):
        conv_layer_name = 'conv2d_%d' %i 

        conv_layer = model.get_layer(conv_layer_name)
        
        filters = conv_layer.filters
        k_size = conv_layer.kernel_size[0]
        in_channels = conv_layer.input_shape[-1]
        
        weights = np.load(str(weights_path / (tabular[conv_layer_name]+'.weight.npy')))
        weights = reshape_weights(weights,filters,k_size,in_channels,torch_weights_channels_first)
        bias = np.load(str(weights_path / (tabular[conv_layer_name]+'.bias.npy')))
        assert bias.shape == (filters,), "Dimension missmatch reading bias with shape {} expected {}".format(bias.shape,(filters,))
        scale = np.load(str(weights_path / (tabular[conv_layer_name]+'.activation_shift.npy')))
        assert scale.shape == (1,), "Dimension missmatch reading bias with shape {} expected {}".format(bias.shape,(1,))
        scale = scale[0]
        conv_layer.set_weights([weights, bias])
        conv_layer.out_shfit = scale


def read_class_names(class_file_name):
    names = {}
    with open(class_file_name, 'r') as data:
        for ID, name in enumerate(data):
            names[ID] = name.strip('\n')
    return names

def load_config(model='yolov4',tiny=False):
    if tiny:
        STRIDES = np.array(cfg.YOLO.STRIDES_TINY)
        ANCHORS = get_anchors(cfg.YOLO.ANCHORS_TINY, tiny)
        XYSCALE = cfg.YOLO.XYSCALE_TINY if model == 'yolov4' else [1, 1]
    else:
        STRIDES = np.array(cfg.YOLO.STRIDES)
        if model == 'yolov4':
            ANCHORS = get_anchors(cfg.YOLO.ANCHORS, tiny)
        elif model == 'yolov3':
            ANCHORS = get_anchors(cfg.YOLO.ANCHORS_V3, tiny)
        else:
            Exception('Model not supported')
            ANCHORS = get_anchors(cfg.YOLO.ANCHORS_V3, tiny)
        XYSCALE = cfg.YOLO.XYSCALE if model == 'yolov4' else [1, 1, 1]
    NUM_CLASS = len(read_class_names(cfg.YOLO.CLASSES))

    return STRIDES, ANCHORS, NUM_CLASS, XYSCALE

def get_anchors(anchors_path, tiny=False):
    anchors = np.array(anchors_path)
    if tiny:
        return anchors.reshape(2, 3, 2)
    else:
        return anchors.reshape(3, 3, 2)

def image_preprocess(image, target_size, gt_boxes=None):

    ih, iw    = target_size
    h,  w, _  = image.shape

    scale = min(iw/w, ih/h)
    nw, nh  = int(scale * w), int(scale * h)
    image_resized = cv2.resize(image, (nw, nh))

    image_paded = np.full(shape=[ih, iw, 3], fill_value=128.0)
    dw, dh = (iw - nw) // 2, (ih-nh) // 2
    image_paded[dh:nh+dh, dw:nw+dw, :] = image_resized
    image_paded = image_paded / 255.

    if gt_boxes is None:
        return image_paded

    else:
        gt_boxes[:, [0, 2]] = gt_boxes[:, [0, 2]] * scale + dw
        gt_boxes[:, [1, 3]] = gt_boxes[:, [1, 3]] * scale + dh
        return image_paded, gt_boxes

def draw_bbox(image, bboxes, classes=read_class_names(cfg.YOLO.CLASSES), show_label=True):
    num_classes = len(classes)
    image_h, image_w, _ = image.shape
    hsv_tuples = [(1.0 * x / num_classes, 1., 1.) for x in range(num_classes)]
    colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
    colors = list(map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)), colors))

    random.seed(0)
    random.shuffle(colors)
    random.seed(None)

    out_boxes, out_scores, out_classes, num_boxes = bboxes
    for i in range(num_boxes[0]):
        if int(out_classes[0][i]) < 0 or int(out_classes[0][i]) > num_classes: continue
        coor = out_boxes[0][i]
        coor[0] = int(coor[0] * image_h)
        coor[2] = int(coor[2] * image_h)
        coor[1] = int(coor[1] * image_w)
        coor[3] = int(coor[3] * image_w)

        fontScale = 0.5
        score = out_scores[0][i]
        class_ind = int(out_classes[0][i])
        bbox_color = colors[class_ind]
        bbox_thick = int(0.6 * (image_h + image_w) / 600)
        c1, c2 = (coor[1], coor[0]), (coor[3], coor[2])
        cv2.rectangle(image, c1, c2, bbox_color, bbox_thick)

        if show_label:
            bbox_mess = '%s: %.2f' % (classes[class_ind], score)
            t_size = cv2.getTextSize(bbox_mess, 0, fontScale, thickness=bbox_thick // 2)[0]
            c3 = (c1[0] + t_size[0], c1[1] - t_size[1] - 3)
            cv2.rectangle(image, c1, (np.float32(c3[0]), np.float32(c3[1])), bbox_color, -1) #filled

            cv2.putText(image, bbox_mess, (c1[0], np.float32(c1[1] - 2)), cv2.FONT_HERSHEY_SIMPLEX,
                        fontScale, (0, 0, 0), bbox_thick // 2, lineType=cv2.LINE_AA)
    return image

def draw_bbox_nms(image, bboxes, classes=read_class_names(cfg.YOLO.CLASSES), show_label=True):
    num_classes = len(classes)
    image_h, image_w, _ = image.shape
    hsv_tuples = [(1.0 * x / num_classes, 1., 1.) for x in range(num_classes)]
    colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
    colors = list(map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)), colors))

    random.seed(0)
    random.shuffle(colors)
    random.seed(None)

    out_boxes, out_scores, out_classes = bboxes
    for i in range(len(out_boxes)):
        if int(out_classes[i]) < 0 or int(out_classes[i]) > num_classes: continue
        coor = out_boxes[i]
        coor[0] = int(coor[0] * image_h)
        coor[2] = int(coor[2] * image_h)
        coor[1] = int(coor[1] * image_w)
        coor[3] = int(coor[3] * image_w)
        coor = coor.astype(np.int32)

        fontScale = 0.5
        score = out_scores[i]
        class_ind = int(out_classes[i])
        bbox_color = colors[class_ind]
        bbox_thick = int(0.6 * (image_h + image_w) / 600)
        c1, c2 = (coor[1], coor[0]), (coor[3], coor[2])
        cv2.rectangle(image, c1, c2, bbox_color, bbox_thick)

        if show_label:
            bbox_mess = '%s: %.2f' % (classes[class_ind], score)
            t_size = cv2.getTextSize(bbox_mess, 0, fontScale, thickness=bbox_thick // 2)[0]
            c3 = (c1[0] + t_size[0], c1[1] - t_size[1] - 3)
            cv2.rectangle(image, c1, (np.float32(c3[0]), np.float32(c3[1])), bbox_color, -1) #filled

            cv2.putText(image, bbox_mess, (c1[0], np.float32(c1[1] - 2)), cv2.FONT_HERSHEY_SIMPLEX,
                        fontScale, (0, 0, 0), bbox_thick // 2, lineType=cv2.LINE_AA)
    return image

def bbox_iou(bboxes1, bboxes2):
    """
    @param bboxes1: (a, b, ..., 4)
    @param bboxes2: (A, B, ..., 4)
        x:X is 1:n or n:n or n:1
    @return (max(a,A), max(b,B), ...)
    ex) (4,):(3,4) -> (3,)
        (2,1,4):(2,3,4) -> (2,3)
    """
    bboxes1_area = bboxes1[..., 2] * bboxes1[..., 3]
    bboxes2_area = bboxes2[..., 2] * bboxes2[..., 3]

    bboxes1_coor = np.concatenate(
        [
            bboxes1[..., :2] - bboxes1[..., 2:] * 0.5,
            bboxes1[..., :2] + bboxes1[..., 2:] * 0.5,
        ],
        axis=-1,
    )
    bboxes2_coor = np.concatenate(
        [
            bboxes2[..., :2] - bboxes2[..., 2:] * 0.5,
            bboxes2[..., :2] + bboxes2[..., 2:] * 0.5,
        ],
        axis=-1,
    )

    left_up = np.maximum(bboxes1_coor[..., :2], bboxes2_coor[..., :2])
    right_down = np.minimum(bboxes1_coor[..., 2:], bboxes2_coor[..., 2:])

    inter_section = np.maximum(right_down - left_up, 0.0)
    inter_area = inter_section[..., 0] * inter_section[..., 1]

    union_area = bboxes1_area + bboxes2_area - inter_area

    iou = np.divide(inter_area,union_area,out=np.zeros_like(inter_area), where=union_area!=0,casting='unsafe')

    return iou

def nms(bboxes, iou_threshold, sigma=0.3, method='nms'):
    """
    :param bboxes: (xmin, ymin, xmax, ymax, score, class)

    Note: soft-nms, https://arxiv.org/pdf/1704.04503.pdf
          https://github.com/bharatsingh430/soft-nms
    """
    classes_in_img = list(set(bboxes[:, 5]))
    best_bboxes = []

    for clss in classes_in_img:
        clss_mask = (bboxes[:, 5] == clss)
        clss_bboxes = bboxes[clss_mask]

        while len(clss_bboxes) > 0:
            max_ind = np.argmax(clss_bboxes[:, 4])
            best_bbox = clss_bboxes[max_ind]
            best_bboxes.append(best_bbox)
            clss_bboxes = np.concatenate([clss_bboxes[: max_ind], clss_bboxes[max_ind + 1:]]) # selcting all boxes except of best box
            iou = bbox_iou(best_bbox[np.newaxis, :4], clss_bboxes[:, :4])
            weight = np.ones((len(iou),), dtype=np.float32)

            assert method in ['nms', 'soft-nms']

            if method == 'nms':
                iou_mask = iou > iou_threshold
                weight[iou_mask] = 0.0

            if method == 'soft-nms':
                weight = np.exp(-(1.0 * iou ** 2 / sigma))

            clss_bboxes[:, 4] = clss_bboxes[:, 4] * weight
            score_mask = clss_bboxes[:, 4] > 0.
            clss_bboxes = clss_bboxes[score_mask]

    return best_bboxes

def filter_boxes_np(box_xywh, scores, score_threshold=0.4, input_shape = [416,416]):
    scores_max = np.max(scores, axis=-1)

    mask = scores_max >= score_threshold
    class_boxes = box_xywh[mask]
    pred_conf = scores[mask]
    class_boxes = np.reshape(class_boxes, [np.shape(scores)[0], -1, np.shape(class_boxes)[-1]])
    pred_conf = np.reshape(pred_conf, [np.shape(scores)[0], -1, np.shape(pred_conf)[-1]])

    box_xy = class_boxes[...,:2]
    box_wh = class_boxes[...,2:]

    input_shape = np.array(input_shape,dtype=np.float32)

    box_yx = box_xy[..., ::-1]
    box_hw = box_wh[..., ::-1]

    box_mins = (box_yx - (box_hw / 2.)) / input_shape
    box_maxes = (box_yx + (box_hw / 2.)) / input_shape
    boxes = np.concatenate([
        box_mins[..., 0:1],  # y_min
        box_mins[..., 1:2],  # x_min
        box_maxes[..., 0:1],  # y_max
        box_maxes[..., 1:2]  # x_max
    ], axis=-1)
    # return tf.concat([boxes, pred_conf], axis=-1)
    return (boxes, pred_conf)
