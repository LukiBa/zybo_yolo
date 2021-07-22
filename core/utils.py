import cv2
import random
import colorsys
import numpy as np

def read_class_names(class_file_name):
    names = {}
    with open(class_file_name, 'r') as data:
        for ID, name in enumerate(data):
            names[ID] = name.strip('\n')
    return names

def draw_bbox_nms(image, bboxes, classes, show_label=True):
    num_classes = len(classes)
    image_h, image_w, _ = image.shape
    hsv_tuples = [(1.0 * x / num_classes, 1., 1.) for x in range(num_classes)]
    colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
    colors = list(map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)), colors))

    random.seed(0)
    random.shuffle(colors)
    random.seed(None)

    out_boxes = bboxes[:,:4]
    out_scores = bboxes[:,4]
    out_classes = bboxes[:,5]
    for i in range(len(out_boxes)):
        if int(out_classes[i]) < 0 or int(out_classes[i]) > num_classes: continue
        coor = out_boxes[i]
        # coor[0] = int(coor[0] * image_h)
        # coor[2] = int(coor[2] * image_h)
        # coor[1] = int(coor[1] * image_w)
        # coor[3] = int(coor[3] * image_w)
        coor = coor.astype(np.int32)

        fontScale = 0.5
        score = out_scores[i]
        class_ind = int(out_classes[i])
        bbox_color = colors[class_ind]
        bbox_thick = int(0.6 * (image_h + image_w) / 600)
        c1, c2 = (coor[0], coor[1]), (coor[2], coor[3])
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


def xywh2xyxy(x):
    # Transform box coordinates from [x, y, w, h] to [x1, y1, x2, y2] (where xy1=top-left, xy2=bottom-right)
    y = np.zeros_like(x)
    y[:, 0] = x[:, 0] - x[:, 2] / 2  # top left x
    y[:, 1] = x[:, 1] - x[:, 3] / 2  # top left y
    y[:, 2] = x[:, 0] + x[:, 2] / 2  # bottom right x
    y[:, 3] = x[:, 1] + x[:, 3] / 2  # bottom right y
    return y

def filter_boxes(x, conf_thres=0.1):
    x = x[x[:, 4] > conf_thres]
    if not x.shape[0]:
        return x

    # Compute conf
    x[..., 5:] *= x[..., 4:5]  # conf = obj_conf * cls_conf

    boxes = xywh2xyxy(x[:, :4])
    classes = np.argmax(x[:,5:],axis=-1)
    pred_conf = np.take_along_axis(x[:,5:], np.expand_dims(classes, axis=-1), axis=-1) # get values by index array 

    return (boxes, pred_conf, classes)

def nms(boxes,pred_conf, classes, iou_threshold, sigma=0.3, score=0.1, method='nms'):
    """
    :param bboxes: (xmin, ymin, xmax, ymax, score, class)

    Note: soft-nms, https://arxiv.org/pdf/1704.04503.pdf
          https://github.com/bharatsingh430/soft-nms
    """
    classes_in_img = list(set(classes))
    bboxes = np.concatenate((boxes,pred_conf,classes.reshape(classes.size,1)),axis=-1)
    best_bboxes = []
    

    for clss in classes_in_img:
        clss_mask = (classes == clss)
        clss_bboxes = bboxes[clss_mask]

        while len(clss_bboxes) > 0:
            max_ind = np.argmax(clss_bboxes[:,4])
            best_bbox = clss_bboxes[max_ind]
            clss_bboxes = np.concatenate([clss_bboxes[: max_ind], clss_bboxes[max_ind + 1:]]) # selcting all boxes except of best box
            iou = bbox_iou(best_bbox[np.newaxis, :4].copy(), clss_bboxes[:, :4])
            weight = np.ones((len(iou),), dtype=np.float32)

            if method == 'nms':
                iou_mask = iou > iou_threshold
                weight[iou_mask] = 0.0

            elif method == 'soft-nms':
                weight = np.exp(-(1.0 * iou ** 2 / sigma))
                
            elif method == 'merge':
                iou_mask = iou > iou_threshold
                weight[iou_mask] = 0.0
                iou_boxes = np.concatenate((clss_bboxes,iou.reshape(iou.size,1)),axis=-1)
                merge_boxes = iou_boxes[iou_mask]
                for i in range(merge_boxes.shape[0]):
                   best_bbox[:4] -= (best_bbox[:4]- merge_boxes[i,:4])*(0.5*merge_boxes[i,4]/best_bbox[4])
                   best_bbox[4] += merge_boxes[i,4]*merge_boxes[i,6]*sigma
            else:
               raise NotImplementedError('Non max surpression method :"'+ method + '" is not implemented.')
                   
            clss_bboxes[:, 4] = clss_bboxes[:, 4] * weight
            score_mask = clss_bboxes[:, 4] > 0.
            clss_bboxes = clss_bboxes[score_mask]
            if best_bbox[4] > score:
                best_bboxes.append(best_bbox)

    return np.array(best_bboxes)


class YoloLayer():
    def __init__(self, anchors : np.ndarray, num_classes : int, stride : int):
        self.anchors = anchors
        self.stride = stride  # layer stride
        self.na = len(anchors)  # number of anchors (3)
        self.nc = num_classes  # number of classes (80)
        self.no = num_classes + 5  # number of outputs (85)
        self.nx, self.ny, self.ng = 0, 0, 0  # initialize number of x, y gridpoints
        self.anchor_vec = self.anchors / self.stride
        self.anchor_wh = self.anchor_vec.reshape(1, self.na, 1, 1, 2)
        self.grid = None

    def create_grids(self, ng=(13,13)):
        self.ny, self.nx = ng  # y and x grid size
        self.ng = ng

        xv, yv = np.meshgrid(np.arange(self.nx), np.arange(self.ny))     
        self.grid = np.stack((xv,yv),axis=-1).reshape(1,1,self.ny,self.ny,2).astype(np.float32)   
    
    def sigmoid(self,x):
        return 1.0 / (1.0 + np.exp(-x))
    
    def __call__(self,p):
        if self.grid == None:
            _, ny, nx = p.shape  # 255, 13, 13
            self.create_grids((ny,nx))
        
        io = np.transpose(p.reshape(self.na, self.no, self.ny, self.nx),(0,2,3,1)).copy('C')

        io[..., :2] = self.sigmoid(io[..., :2]) + self.grid  # xy
        io[..., 2:4] = np.exp(io[..., 2:4]) * self.anchor_wh  # wh yolo method
        io[..., :4] *= self.stride
        io[..., 4:] = self.sigmoid(io[..., 4:])
        return io.reshape(-1, self.no) # view [1, 3, 13, 13, 85] as [1, 507, 85]        