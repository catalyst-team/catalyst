from mean_average_precision import MetricBuilder
import numpy as np
import torch
import torch.nn.functional as F

from catalyst.core.callback import Callback, CallbackOrder

from .utils import change_box_order, nms_filter


def process_ssd_output(
    predicted_bboxes,
    predicted_scores,
    gt_bboxes,
    gt_labels,
    width=300,
    height=300,
    ignore_class=0,
    iou_threshold=0.5,
):
    """Generate bbox and classes from SSD model outputs.

    Args:
        predicted_bboxes (torch.Tensor): predicted bounding boxes,
            expected shapes [batch, num anchors, 4].
        predicted_scores (torch.Tensor): predicted class scores,
            expected shapes [batch, num anchors, num classes + 1].
        gt_bboxes (torch.Tensor): ground truth bounding boxes,
            expected shapes [batch, num anchors, 4].
        gt_labels (torch.Tensor): ground truth bounding box labels,
            expected shape [batch, num anchors].
        width (int): model input image width.
            Default is ``300``.
        height (int): model input image height.
            Default is ``300``.
        ignore_class (int): index of background class, this class should be ignored.
            Default is ``0``.
        iou_threshold (float): IoU threshold to use in NMS.
            Default is ``0.5``.

    Yields:
        predicted sample (np.ndarray) and ground truth sample (np.ndarray)
    """
    batch_size = predicted_bboxes.size(0)

    pred_boxes = predicted_bboxes.detach().cpu().numpy()
    pred_confidence, pred_cls = torch.max(F.softmax(predicted_scores, dim=-1), dim=-1)
    pred_confidence = pred_confidence.detach().cpu().numpy()
    pred_cls = pred_cls.detach().cpu().numpy()

    for i in range(batch_size):
        # build predictions
        sample_bboxes = change_box_order(pred_boxes[i], "xywh2xyxy")
        sample_bboxes = (sample_bboxes * (width, height, width, height)).astype(np.int32)
        sample_bboxes, sample_classes, sample_confs = nms_filter(
            sample_bboxes, pred_cls[i], pred_confidence[i], iou_threshold=iou_threshold
        )
        pred_sample = np.concatenate(
            [sample_bboxes, sample_classes[:, None], sample_confs[:, None]], -1
        )

        # build ground truth
        sample_target_classes = gt_labels[i].detach().cpu().numpy()
        sample_target_bboxes = gt_bboxes[i, :].detach().cpu().numpy()
        # drop empty bboxes
        mask = sample_target_classes != ignore_class
        sample_target_bboxes = sample_target_bboxes[mask, :]
        sample_target_classes = sample_target_classes[mask]
        # convert to required format
        sample_target_bboxes = change_box_order(sample_target_bboxes, "xywh2xyxy")
        sample_target_bboxes = (sample_target_bboxes * (width, height, width, height)).astype(
            np.int32
        )
        gt_sample = np.zeros((sample_target_classes.shape[0], 7), dtype=np.float32)
        gt_sample[:, :4] = sample_target_bboxes
        gt_sample[:, 4] = sample_target_classes

        yield pred_sample, gt_sample


def pred2box(heatmap, regr, threshold=0.5, scale=4, input_size=512):
    """Convert model output to bounding boxes.

    Args:
        heatmap (np.ndarray): center heatmap, expected matrix with shapes [N, N].
        regr (np.ndarray): width and height coordinates, expected matrix with shapes [2, N, N]
        threshold (float): score threshold.
            If ``None`` then will be ignored filtering.
            Default is ``None``.
        scale (int): output scale, resulting coordinates will be multiplied by that constant.
            Default is ``4``.
        input_size (int): model input size.
            Default is ``512``.

    Returns:
        bounding boxes (np.ndarray with shape [M, 4]) and scores (np.ndarray with shape [M])
    """
    cy, cx = np.where(heatmap > threshold)
    boxes, scores = np.empty((len(cy), 4), dtype=int), np.zeros(len(cx), dtype=float)
    for i, (x, y) in enumerate(zip(cx, cy)):
        scores[i] = heatmap[y, x]
        # x, y in segmentation scales -> upscale outputs
        sx, sy = int(x * scale), int(y * scale)
        w, h = (regr[:, y, x] * input_size).astype(int)
        boxes[i, 0] = sx - w // 2
        boxes[i, 1] = sy - h // 2
        boxes[i, 2] = sx + w // 2
        boxes[i, 3] = sy + h // 2
    return np.array(boxes), np.array(scores)


def process_centernet_output(
    predicted_heatmap,  # logits
    predicted_regression,
    gt_boxes,
    gt_labels,
    confidence_threshold=0.5,
    iou_threshold=0.5,
):
    """Generate bbox and classes from CenterNet model outputs.

    Args:
        predicted_heatmap (torch.Tensor): predicted center heatmap logits,
            expected shapes [batch, height, width, num classes].
        predicted_regression (torch.Tensor): predicted HW regression,
            expected shapes [batch, height, width, 2].
        gt_boxes (List[torch.Tensor]): list with sample bounding boxes.
        gt_labels (List[torch.Tensor]): list with sample bounding box labels.
        confidence_threshold (float): confidence threshold,
            proposals with lover values than threshold will be ignored.
            Default is ``0.5``.
        iou_threshold (float): IoU threshold to use in NMS.
            Default is ``0.5``.

    Yields:
        predicted sample (np.ndarray) and ground truth sample (np.ndarray)
    """
    batch_size = predicted_heatmap.size(0)

    hm = predicted_heatmap.sigmoid()
    pooled = F.max_pool2d(hm, kernel_size=(3, 3), stride=1, padding=1)
    hm *= torch.logical_and(hm >= confidence_threshold, pooled >= confidence_threshold).float()

    hm_numpy = hm.detach().cpu().numpy()
    reg_numpy = predicted_regression.detach().cpu().numpy()

    for i in range(batch_size):
        sample_boxes = []
        sample_classes = []
        sample_scores = []
        for cls_idx in range(hm_numpy.shape[1]):
            # build predictions
            cls_boxes, cls_scores = pred2box(
                hm_numpy[i, cls_idx], reg_numpy[i], threshold=0, scale=4, input_size=512
            )

            # skip empty label predictions
            if cls_scores.shape[0] == 0:
                continue

            cls_boxes = cls_boxes / 512.0

            cls_boxes, cls_classes, cls_scores = nms_filter(
                cls_boxes,
                np.full(len(cls_scores), cls_idx),
                cls_scores,
                iou_threshold=iou_threshold,
            )
            sample_boxes.append(cls_boxes)
            sample_classes.append(cls_classes)
            sample_scores.append(cls_scores)
        # skip empty predictions
        if len(sample_boxes) == 0:
            continue

        sample_boxes = np.concatenate(sample_boxes, 0)
        sample_classes = np.concatenate(sample_classes, 0)
        sample_scores = np.concatenate(sample_scores, 0)

        pred_sample = np.concatenate(
            [sample_boxes, sample_classes[:, None], sample_scores[:, None]], -1
        )
        pred_sample = pred_sample.astype(np.float32)

        sample_gt_bboxes = gt_boxes[i].detach().cpu()
        sample_gt_classes = gt_labels[i].detach().cpu()
        gt_sample = np.zeros((sample_gt_classes.shape[0], 7), dtype=np.float32)
        gt_sample[:, :4] = sample_gt_bboxes
        gt_sample[:, 4] = sample_gt_classes

        yield pred_sample, gt_sample


class DetectionMeanAveragePrecision(Callback):
    """Compute mAP for Object Detection task."""

    def __init__(
        self,
        num_classes=1,
        metric_key="mAP",
        output_type="ssd",
        iou_threshold=0.5,
        confidence_threshold=0.5,
    ):
        """
        Args:
            num_classes (int): Number of classes.
                Default is ``1``.
            metric_key (str): name of a metric.
                Default is ``"mAP"``.
            output_type (str): model output type. Valid values are ``"ssd"`` or ``"centernet"``.
                Default is ``"ssd"``.
            iou_threshold (float): IoU threshold to use in NMS.
                Default is ``0.5``.
            confidence_threshold (float): confidence threshold,
                proposals with lover values than threshold will be ignored.
                Default is ``0.5``.
        """
        super().__init__(order=CallbackOrder.Metric)
        assert output_type in ("ssd", "centernet")

        self.num_classes = num_classes
        self.metric_key = metric_key
        self.output_type = output_type
        self.iou_threshold = iou_threshold
        self.confidence_threshold = confidence_threshold

        self.metric_fn = MetricBuilder.build_evaluation_metric(
            "map_2d", async_mode=False, num_classes=num_classes
        )

    def on_loader_start(self, runner: "IRunner"):  # noqa: D102, F821
        if not runner.is_valid_loader:
            return
        self.metric_fn.reset()

    def on_batch_end(self, runner: "IRunner"):  # noqa: D102, F821
        if not runner.is_valid_loader:
            return

        if self.output_type == "ssd":
            p_box = runner.batch["predicted_bboxes"]
            gt_box = runner.batch["bboxes"]
            p_scores = runner.batch["predicted_scores"]
            gt_labels = runner.batch["labels"]
            for predicted_sample, ground_truth_sample in process_ssd_output(
                p_box, p_scores, gt_box, gt_labels, iou_threshold=self.iou_threshold
            ):
                self.metric_fn.add(predicted_sample, ground_truth_sample)
        elif self.output_type == "centernet":
            p_heatmap = runner.batch["predicted_heatmap"]
            gt_box = runner.batch["bboxes"]
            p_regression = runner.batch["predicted_regression"]
            gt_labels = runner.batch["labels"]
            for predicted_sample, ground_truth_sample in process_centernet_output(
                p_heatmap,
                p_regression,
                gt_box,
                gt_labels,
                iou_threshold=self.iou_threshold,
                confidence_threshold=self.confidence_threshold,
            ):
                self.metric_fn.add(predicted_sample, ground_truth_sample)

    def on_loader_end(self, runner: "IRunner"):  # noqa: D102, F821
        if not runner.is_valid_loader:
            return
        map_value = self.metric_fn.value()["mAP"]
        runner.loader_metrics[self.metric_key] = map_value
