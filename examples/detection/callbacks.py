from mean_average_precision import MetricBuilder
import numpy as np
import torch
import torch.nn.functional as F

from catalyst.core.callback import Callback, CallbackNode, CallbackOrder

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


class DetectionMeanAveragePrecision(Callback):
    def __init__(self, num_classes=1, metric_key="mAP", output_type="ssd", iou_threshold=0.5):
        super().__init__(order=CallbackOrder.Metric, node=CallbackNode.Master)
        assert output_type in ("ssd",)

        self.num_classes = num_classes
        self.metric_key = metric_key
        self.output_type = output_type
        self.iou_threshold = iou_threshold

        self.metric_fn = MetricBuilder.build_evaluation_metric(
            "map_2d", async_mode=False, num_classes=num_classes
        )

    def on_loader_start(self, runner: "IRunner"):
        if not runner.is_valid_loader:
            return
        self.metric_fn.reset()

    def on_batch_end(self, runner: "IRunner"):
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

    def on_loader_end(self, runner: "IRunner"):
        if not runner.is_valid_loader:
            return
        map_value = self.metric_fn.value()["mAP"]
        runner.loader_metrics[self.metric_key] = map_value
