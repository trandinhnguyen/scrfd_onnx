import numpy as np


def nms(boxes: np.ndarray, order: np.ndarray, nms_thresh: float) -> list[int]:
    """Perform non-maximum suppression on bounding boxes

    Args:
        - boxes: bounding boxes have shape (:, 4)
        - order: sorted indices
        - nms_threshold: iou threshold
        
    Returns:
        - list of indices
    """
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]
    areas = (x2 - x1) * (y2 - y1)

    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)  # always keep the first element

        # Compute ious
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0.0, xx2 - xx1)
        h = np.maximum(0.0, yy2 - yy1)
        intersection = w * h
        iou = intersection / (areas[i] + areas[order[1:]] - intersection + 1e-8)

        inds = np.where(iou <= nms_thresh)[0]  # get rid of boxes whose iou is large
        order = order[inds + 1]  # increase indices by 1 to eliminate current box
    return keep
