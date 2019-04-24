from typing import Callable, List, Dict
import numpy as np

from torchnet.meter import AUCMeter, ConfusionMeter
from sklearn.metrics import confusion_matrix as confusion_matrix_fn
from tensorboardX import SummaryWriter

import torch
from catalyst.dl import metrics
from catalyst.dl.state import RunnerState

from .core import Callback
from .loggers import TensorboardLogger
from .utils import plot_confusion_matrix, render_figure_to_tensor


class MetricCallback(Callback):
    """
    A callback that returns single metric on `state.on_batch_end`
    """

    def __init__(
        self,
        prefix: str,
        metric_fn: Callable,
        input_key: str = "targets",
        output_key: str = "logits",
        **metric_params
    ):
        self.prefix = prefix
        self.metric_fn = metric_fn
        self.input_key = input_key
        self.output_key = output_key
        self.metric_params = metric_params

    def on_batch_end(self, state: RunnerState):
        outputs = state.output[self.output_key]
        targets = state.input[self.input_key]
        metric = self.metric_fn(outputs, targets, **self.metric_params)
        state.metrics.add_batch_value(name=self.prefix, value=metric)


class MultiMetricCallback(Callback):
    """
    A callback that returns multiple metrics on `state.on_batch_end`
    """

    def __init__(
        self,
        prefix: str,
        metric_fn: Callable,
        list_args: List,
        input_key: str = "targets",
        output_key: str = "logits",
        **metric_params
    ):
        self.prefix = prefix
        self.metric_fn = metric_fn
        self.list_args = list_args
        self.input_key = input_key
        self.output_key = output_key
        self.metric_params = metric_params

    def on_batch_end(self, state: RunnerState):
        outputs = state.output[self.output_key]
        targets = state.input[self.input_key]

        metrics_ = self.metric_fn(
            outputs, targets, self.list_args, **self.metric_params
        )

        batch_metrics = {}
        for arg, metric in zip(self.list_args, metrics_):
            if isinstance(arg, int):
                key = f"{self.prefix}{arg:02}"
            else:
                key = f"{self.prefix}_{arg}"
            batch_metrics[key] = metric
        state.metrics.add_batch_value(metrics_dict=batch_metrics)


class DiceCallback(MetricCallback):
    """
    Dice metric callback.
    """

    def __init__(
        self,
        input_key: str = "targets",
        output_key: str = "logits",
        prefix: str = "dice",
        eps: float = 1e-7,
        activation: str = "sigmoid"
    ):
        """
        :param input_key: input key to use for dice calculation;
            specifies our `y_true`.
        :param output_key: output key to use for dice calculation;
            specifies our `y_pred`.
        """
        super().__init__(
            prefix=prefix,
            metric_fn=metrics.dice,
            input_key=input_key,
            output_key=output_key,
            eps=eps,
            activation=activation
        )


class IouCallback(MetricCallback):
    """
    IoU (Jaccard) metric callback.
    """

    def __init__(
        self,
        input_key: str = "targets",
        output_key: str = "logits",
        prefix: str = "iou",
        mode: str = "hard",
        eps: float = 1e-7,
        threshold: float = 0.5,
        activation: str = "sigmoid",
    ):
        """
        Args:
            input_key (str): input key to use for iou calculation
                specifies our ``y_true``.
            output_key (str): output key to use for iou calculation;
                specifies our ``y_pred``
            prefix (str): key to store in logs
            mode (str): one of ``['hard', 'soft']`` to calculate IoU
            eps (float): epsilon to avoid zero division
            threshold (float): threshold for outputs binarization
            activation (str): An torch.nn activation applied to the outputs.
                Must be one of ['none', 'sigmoid', 'softmax2d']
        """
        if mode == "hard":
            metric_fn = metrics.iou
        elif mode == "soft":
            metric_fn = metrics.soft_iou
        else:
            raise ValueError(
                f"Mode must be one of ['hard', 'soft'], got {mode}."
            )

        super().__init__(
            prefix=prefix,
            metric_fn=metric_fn,
            input_key=input_key,
            output_key=output_key,
            eps=eps,
            threshold=threshold,
            activation=activation
        )


JaccardCallback = IouCallback


class F1ScoreCallback(MetricCallback):
    """
    F1 score metric callback.
    """

    def __init__(
        self,
        input_key: str = "targets",
        output_key: str = "logits",
        prefix: str = "f1_score",
        beta: float = 1,
        eps: float = 1e-7,
        threshold: float = 0.5,
        activation: str = "sigmoid"
    ):
        """
        Args:
            input_key (str): input key to use for iou calculation
                specifies our ``y_true``.
            output_key (str): output key to use for iou calculation;
                specifies our ``y_pred``
            prefix (str): key to store in logs
            beta (float): beta param for f_score
            eps (float): epsilon to avoid zero division
            threshold (float): threshold for outputs binarization
            activation (str): An torch.nn activation applied to the outputs.
                Must be one of ['none', 'sigmoid', 'softmax2d']
        """

        super().__init__(
            prefix=prefix,
            metric_fn=metrics.f_score,
            input_key=input_key,
            output_key=output_key,
            beta=beta,
            eps=eps,
            threshold=threshold,
            activation=activation
        )


class AccuracyCallback(MultiMetricCallback):
    """
    Accuracy metric callback.
    """

    def __init__(
        self,
        input_key: str = "targets",
        output_key: str = "logits",
        prefix: str = "accuracy",
        accuracy_args: List[int] = None,
    ):
        """
        :param input_key: input key to use for accuracy calculation;
            specifies our `y_true`.
        :param output_key: output key to use for accuracy calculation;
            specifies our `y_pred`.
        :param accuracy_args: specifies which accuracy@K to log.
            [1] - accuracy
            [1, 3] - accuracy at 1 and 3
            [1, 3, 5] - accuracy at 1, 3 and 5
        """
        super().__init__(
            prefix=prefix,
            metric_fn=metrics.accuracy,
            list_args=accuracy_args or [1],
            input_key=input_key,
            output_key=output_key
        )


class MapKCallback(MultiMetricCallback):
    """
    mAP@k metric callback.
    """

    def __init__(
        self,
        input_key: str = "targets",
        output_key: str = "logits",
        prefix: str = "map",
        map_args: List[int] = None,
    ):
        """
        :param input_key: input key to use for
            calculation mean average accuracy at k;
            specifies our `y_true`.
        :param output_key: output key to use for
            calculation mean average accuracy at k;
            specifies our `y_pred`.
        :param map_args: specifies which map@K to log.
            [1] - map@1
            [1, 3] - map@1 and map@3
            [1, 3, 5] - map@1, map@3 and map@5
        """
        super().__init__(
            prefix=prefix,
            metric_fn=metrics.mean_average_accuracy,
            list_args=map_args or [1],
            input_key=input_key,
            output_key=output_key
        )


class AUCCallback(Callback):
    def __init__(
        self,
        input_key: str = "targets",
        output_key: str = "logits",
        prefix: str = "auc",
        class_names: List[str] = None,
        num_classes: int = 1
    ):
        self.prefix = prefix
        self.input_key = input_key
        self.output_key = output_key

        self.class_names = class_names
        self.num_classes = num_classes \
            if class_names is None \
            else len(class_names)

        assert self.num_classes is not None

        self.auc_meters = [AUCMeter() for _ in range(self.num_classes)]

    def _reset_stats(self):
        for auc_meter in self.auc_meters:
            auc_meter.reset()

    def on_loader_start(self, state):
        self._reset_stats()

    def on_batch_end(self, state: RunnerState):
        logits: torch.Tensor = state.output[self.output_key].detach().float()
        targets: torch.Tensor = state.input[self.input_key].detach().float()
        probabilities: torch.Tensor = torch.sigmoid(logits)

        if self.num_classes == 1 and len(probabilities.shape) == 1:
            self.auc_meters[0].add(probabilities, targets)
        else:
            for i in range(self.num_classes):
                self.auc_meters[i].add(probabilities[:, i], targets[:, i])

    def on_loader_end(self, state: RunnerState):
        areas = []

        for i, auc_meter in enumerate(self.auc_meters):
            area, _, _ = auc_meter.value()
            area = float(area)
            postfix = self.class_names[i] \
                if self.class_names is not None \
                else str(i)
            metric_name = f"{self.prefix}/class_{postfix}"
            state.metrics.epoch_values[state.loader_name][metric_name] = area
            areas.append(area)

        area = float(np.mean(areas))
        metric_name = f"{self.prefix}/_mean"
        state.metrics.epoch_values[state.loader_name][metric_name] = area

        self._reset_stats()


class ConfusionMatrixCallback(Callback):
    def __init__(
        self,
        input_key: str = "targets",
        output_key: str = "logits",
        prefix: str = "confusion_matrix",
        version: str = "tnt",
        class_names: List[str] = None,
        num_classes: int = None,
        plot_params: Dict = None
    ):
        self.prefix = prefix
        self.output_key = output_key
        self.input_key = input_key

        assert version in ["tnt", "sklearn"]
        self._version = version
        self._plot_params = plot_params or {}

        self.class_names = class_names
        self.num_classes = num_classes \
            if class_names is None \
            else len(class_names)

        assert self.num_classes is not None
        self._reset_stats()

    @staticmethod
    def _get_tensorboard_logger(state: RunnerState) -> SummaryWriter:
        # @TODO: remove this hack, simplify state
        for logger in state.loggers:
            if isinstance(logger, TensorboardLogger):
                return logger.loggers[state.loader_name]
        raise RuntimeError(
            f"Cannot find Tensorboard logger for loader {state.loader_name}")

    def _reset_stats(self):
        if self._version == "tnt":
            self.confusion_matrix = ConfusionMeter(self.num_classes)
        elif self._version == "sklearn":
            self.outputs = []
            self.targets = []

    def _add_to_stats(self, outputs, targets):
        if self._version == "tnt":
            self.confusion_matrix.add(predicted=outputs, target=targets)
        elif self._version == "sklearn":
            outputs = outputs.cpu().numpy()
            targets = targets.cpu().numpy()

            outputs = np.argmax(outputs, axis=1)

            self.outputs.extend(outputs)
            self.targets.extend(targets)

    def _compute_confusion_matrix(self):
        if self._version == "tnt":
            confusion_matrix = self.confusion_matrix.value()
        elif self._version == "sklearn":
            confusion_matrix = confusion_matrix_fn(
                y_true=self.targets,
                y_pred=self.outputs
            )
        return confusion_matrix

    def _plot_confusion_matrix(
        self,
        logger,
        epoch,
        confusion_matrix,
        class_names=None
    ):
        fig = plot_confusion_matrix(
            confusion_matrix,
            class_names=class_names,
            normalize=True,
            show=False,
            **self._plot_params
        )
        fig = render_figure_to_tensor(fig)
        logger.add_image(f"{self.prefix}/epoch", fig, global_step=epoch)

    def on_loader_start(self, state: RunnerState):
        self._reset_stats()

    def on_batch_end(self, state: RunnerState):
        self._add_to_stats(
            state.output[self.output_key].detach(),
            state.input[self.input_key].detach()
        )

    def on_loader_end(self, state: RunnerState):
        class_names = \
            self.class_names or \
            [str(i) for i in range(self.num_classes)]
        confusion_matrix = self._compute_confusion_matrix()
        logger = self._get_tensorboard_logger(state)
        self._plot_confusion_matrix(
            logger=logger,
            epoch=state.epoch,
            confusion_matrix=confusion_matrix,
            class_names=class_names
        )
