import numpy as np

# YOLO configuration used to display bounding box
yolo_config = {
    'voc': {
        'model': {
            'anchors': [
                0.51926, 1.06255, 1.37274, 1.92085, 1.443, 3.96446, 2.78755,
                4.86559, 5.40293, 5.54216
            ],
            'labels': ["car", "person"]
        },
        'valid': {
            'iou_threshold': 0.5,
            'score_threshold': 0.5
        }
    },
    'widerface': {
        'model': {
            'anchors': [0.90751, 1.49967, 1.63565, 2.43559, 2.93423, 3.88108],
            'labels': ["face"]
        },
        'valid': {
            'iou_threshold': 0.5,
            'score_threshold': 0.5
        }
    }
}


def get_bboxes(potentials, dataset_name, boundaries):
    """
    Generates bounding boxes from akida output to original input image size.

    Args:
        potentials(Numpy.array): akida output potentials
        dataset_name(str): 'widerface' or 'voc'. Used to get anchors.
        boundaries(tuple): x, y, width, and height of input image

    Returns:
        bboxes(list): A list of dicts with bounding boxes coordinates and
        label
    """
    bboxes = []
    h, w, c = potentials.shape
    class_names = yolo_config[dataset_name]['model']['labels']
    n_values = 4 + 1 + len(class_names)
    # Sanity check
    if c % n_values == 0:
        n_boxes = c // n_values
        # Reshape potentials to split channels into values for each box
        potentials = potentials.reshape((h, w, n_boxes, n_values))
        # Inverts width and height
        potentials = potentials.transpose((1, 0, 2, 3))
        # Evaluate boxes
        boxes = decode_snnout(
            potentials, yolo_config[dataset_name]['model']['anchors'],
            len(class_names),
            yolo_config[dataset_name]['valid']['iou_threshold'], 0.3)

        x, y, w, h = boundaries
        for box in boxes:
            # Evaluate box size
            box_w = int((box.xmax - box.xmin) * w)
            box_h = int((box.ymax - box.ymin) * h)
            box_x = x + int(box.xmin * w)
            box_y = y + int((box.ymax) * h)

            bboxes.append({
                'x1': box_x,
                'y1': box_y,
                'x2': box_x + box_w,
                'y2': box_y - box_h,
                'label': class_names[box.get_label()]
            })

    return bboxes


class BoundBox:

    def __init__(self, xmin, ymin, xmax, ymax, c=None, classes=None):
        self.xmin = xmin
        self.ymin = ymin
        self.xmax = xmax
        self.ymax = ymax

        self.c = c
        self.classes = classes

        self.label = -1
        self.score = -1

    def get_label(self):
        if self.label == -1:
            self.label = np.argmax(self.classes)

        return self.label

    def get_score(self):
        if self.score == -1:
            self.score = self.classes[self.get_label()]
        return self.score

    def __repr__(self):
        """
        Helper method for printing the object's values
        :return:
        """
        return "<BoundBox({}, {}, {}, {}, {}, {})>\n".format(
            self.xmin, self.xmax, self.ymin, self.ymax, self.get_label(),
            self.get_score())


def bbox_iou(box1, box2):
    intersect_w = _interval_overlap([box1.xmin, box1.xmax],
                                    [box2.xmin, box2.xmax])
    intersect_h = _interval_overlap([box1.ymin, box1.ymax],
                                    [box2.ymin, box2.ymax])

    intersect = intersect_w * intersect_h

    w1, h1 = box1.xmax - box1.xmin, box1.ymax - box1.ymin
    w2, h2 = box2.xmax - box2.xmin, box2.ymax - box2.ymin

    union = w1 * h1 + w2 * h2 - intersect

    return float(intersect) / union


def decode_snnout(netout,
                  anchors,
                  nb_class,
                  obj_threshold=0.5,
                  nms_threshold=0.3):
    grid_h, grid_w, nb_box = netout.shape[:3]

    boxes = []

    # decode the output by the network
    netout[..., 4] = _sigmoid(netout[..., 4])
    netout[...,
           5:] = netout[..., 4][..., np.newaxis] * _softmax(netout[..., 5:])
    netout[..., 5:] *= netout[..., 5:] > obj_threshold

    for row in range(grid_h):
        for col in range(grid_w):
            for b in range(nb_box):
                # from 4th element onwards are confidence and class classes
                classes = netout[row, col, b, 5:]

                if np.sum(classes) > 0:
                    # first 4 elements are x, y, w, and h
                    x, y, w, h = netout[row, col, b, :4]

                    x = (col + _sigmoid(x)
                        ) / grid_w  # center position, unit: image width
                    y = (row + _sigmoid(y)
                        ) / grid_h  # center position, unit: image height
                    w = anchors[2 * b +
                                0] * np.exp(w) / grid_w  # unit: image width
                    h = anchors[2 * b +
                                1] * np.exp(h) / grid_h  # unit: image height
                    confidence = netout[row, col, b, 4]

                    box = BoundBox(x - w / 2, y - h / 2, x + w / 2, y + h / 2,
                                   confidence, classes)

                    boxes.append(box)

    # suppress non-maximal boxes
    for c in range(nb_class):
        sorted_indices = list(
            reversed(np.argsort([box.classes[c] for box in boxes])))

        for i in range(len(sorted_indices)):
            index_i = sorted_indices[i]

            if boxes[index_i].classes[c] == 0:
                continue
            else:
                for j in range(i + 1, len(sorted_indices)):
                    index_j = sorted_indices[j]

                    if (bbox_iou(boxes[index_i],
                                 boxes[index_j]) >= nms_threshold and
                            c == boxes[index_i].get_label() and
                            c == boxes[index_j].get_label()):
                        boxes[index_i].score = 0

    # remove the boxes which are less likely than a obj_threshold
    boxes = [box for box in boxes if box.get_score() > obj_threshold]

    return boxes


def _interval_overlap(interval_a, interval_b):
    x1, x2 = interval_a
    x3, x4 = interval_b

    if x3 < x1:
        if x4 < x1:
            return 0
        else:
            return min(x2, x4) - x1
    else:
        if x2 < x3:
            return 0
        else:
            return min(x2, x4) - x3


def _sigmoid(x):
    return 1. / (1. + np.exp(-x))


def _softmax(x, axis=-1, t=-100.):
    x = x - np.max(x)

    if np.min(x) < t:
        x = x / np.min(x) * t

    e_x = np.exp(x)

    return e_x / e_x.sum(axis, keepdims=True)
