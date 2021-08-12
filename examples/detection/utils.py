import warnings

import numpy as np
import torch


def change_box_order(boxes, order):
    """Change box order between
    (x_min, y_min, x_max, y_max) <-> (x_center, y_center, width, height).

    Args:
        boxes: (torch.Tensor or np.ndarray) bounding boxes, sized [N,4].
        order: (str) either "xyxy2xywh" or "xywh2xyxy".

    Returns:
        (torch.Tensor) converted bounding boxes, sized [N,4].
    """
    assert order in {"xyxy2xywh", "xywh2xyxy"}
    concat_fn = torch.cat if isinstance(boxes, torch.Tensor) else np.concatenate

    a = boxes[:, :2]
    b = boxes[:, 2:]
    if order == "xyxy2xywh":
        return concat_fn([(a + b) / 2, b - a], 1)
    return concat_fn([a - b / 2, a + b / 2], 1)


def box_nms(bboxes, scores, threshold=0.5):
    """Non maximum suppression.

    Args:
        bboxes (np.ndarray): bounding boxes, sized [N,4].
        scores (np.ndarray): confidence scores, sized [N,].
        threshold (float): overlap threshold.

    Returns:
        keep: (np.ndarray) selected indices.

    Reference:
        https://github.com/rbgirshick/py-faster-rcnn/blob/master/lib/nms/py_cpu_nms.py
    """
    x1, y1 = bboxes[:, 0], bboxes[:, 1]
    x2, y2 = bboxes[:, 2], bboxes[:, 3]

    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = scores.argsort()[::-1]

    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            ovr = inter / (areas[i] + areas[order[1:]] - inter)

        inds = np.where(ovr <= threshold)[0]
        order = order[inds + 1]

    return keep


def nms_filter(bboxes, classes, confidences, iou_threshold=0.5):
    """Filter classes, bboxes, confidences by nms with iou_threshold.

    Args:
        bboxes (np.ndarray): array with bounding boxes, expected shape [N, 4].
        classes (np.ndarray): array with classes, expected shape [N,].
        confidences (np.ndarray): array with class confidence, expected shape [N,].
        iou_threshold (float): IoU threshold to use for filtering.
            Default is ``0.5``.

    Returns:
        filtered bboxes (np.ndarray), classes (np.ndarray), and confidences (np.ndarray)
            where number of records will be equal to some M (M <= N).
    """
    keep_bboxes = []
    keep_classes = []
    keep_confidences = []

    for presented_cls in np.unique(classes):
        mask = classes == presented_cls
        curr_bboxes = bboxes[mask, :]
        curr_classes = classes[mask]
        curr_confs = confidences[mask]

        to_keep = box_nms(curr_bboxes, curr_confs, iou_threshold)

        keep_bboxes.append(curr_bboxes[to_keep, :])
        keep_classes.append(curr_classes[to_keep])
        keep_confidences.append(curr_confs[to_keep])

    return (
        np.concatenate(keep_bboxes),
        np.concatenate(keep_classes),
        np.concatenate(keep_confidences),
    )
