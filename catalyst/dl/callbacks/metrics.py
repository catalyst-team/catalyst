from typing import Callable, List

from catalyst.dl import metrics
from catalyst.dl.state import RunnerState
from .core import Callback


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

    def on_batch_end(self, state):
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


class JaccardCallback(MetricCallback):
    """
    Jaccard metric callback.
    """

    def __init__(
        self,
        input_key: str = "targets",
        output_key: str = "logits",
        prefix: str = "jaccard",
        eps: float = 1e-7
    ):
        """
        :param input_key: input key to use for iou calculation;
            specifies our `y_true`.
        :param output_key: output key to use for iou calculation;
            specifies our `y_pred`
        """
        super().__init__(
            prefix=prefix,
            metric_fn=metrics.jaccard,
            input_key=input_key,
            output_key=output_key,
            eps=eps
        )


class PrecisionCallback(MultiMetricCallback):
    """
    Precision metric callback.
    """

    def __init__(
        self,
        input_key: str = "targets",
        output_key: str = "logits",
        prefix: str = "precision",
        precision_args: List[int] = None,
    ):
        """
        :param input_key: input key to use for precision calculation;
            specifies our `y_true`.
        :param output_key: output key to use for precision calculation;
            specifies our `y_pred`.
        :param precision_args: specifies which precision@K to log.
            [1] - accuracy
            [1, 3] - accuracy and precision@3
            [1, 3, 5] - precision at 1, 3 and 5
        """
        super().__init__(
            prefix=prefix,
            metric_fn=metrics.precision,
            list_args=precision_args or [1, 3, 5],
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
            calculation mean average precision at k;
            specifies our `y_true`.
        :param output_key: output key to use for
            calculation mean average precision at k;
            specifies our `y_pred`.
        :param map_args: specifies which map@K to log.
            [1] - map@1
            [1, 3] - map@1 and map@3
            [1, 3, 5] - map@1, map@3 and map@5
        """
        super().__init__(
            prefix=prefix,
            metric_fn=metrics.mean_average_precision,
            list_args=map_args or [1, 3, 5],
            input_key=input_key,
            output_key=output_key
        )
