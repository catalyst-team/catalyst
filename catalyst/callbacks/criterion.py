# @TODO: rewrite
# from typing import Dict, List, TYPE_CHECKING, Union
#
# from catalyst.callbacks.metric import IBatchMetricCallback
# from catalyst.utils.misc import get_attr
#
# if TYPE_CHECKING:
#     from catalyst.core.runner import IRunner
#
#
# class CriterionCallback(IBatchMetricCallback):
#     """Callback for that measures loss with specified criterion."""
#
#     def __init__(
#         self,
#         input_key: Union[str, List[str], Dict[str, str]] = "targets",
#         output_key: Union[str, List[str], Dict[str, str]] = "logits",
#         prefix: str = "loss",
#         criterion_key: str = None,
#         multiplier: float = 1.0,
#         **metric_kwargs,
#     ):
#         """
#         Args:
#             input_key (Union[str, List[str], Dict[str, str]]): key/list/dict
#                 of keys that takes values from the input dictionary
#                 If '__all__', the whole input will be passed to the criterion
#                 If None, empty dict will be passed to the criterion.
#             output_key (Union[str, List[str], Dict[str, str]]): key/list/dict
#                 of keys that takes values from the input dictionary
#                 If '__all__', the whole output will be passed to the criterion
#                 If None, empty dict will be passed to the criterion.
#             prefix: prefix for metrics and output key for loss
#                 in ``runner.batch_metrics`` dictionary
#             criterion_key: A key to take a criterion in case
#                 there are several of them and they are in a dictionary format.
#             multiplier: scale factor for the output loss.
#         """
#         super().__init__(
#             prefix=prefix,
#             input_key=input_key,
#             output_key=output_key,
#             multiplier=multiplier,
#             **metric_kwargs,
#         )
#         self.criterion_key = criterion_key
#         self._criterion = None
#
#     @property
#     def metric_fn(self):
#         """Criterion function."""
#         return self._criterion
#
#     def on_stage_start(self, runner: "IRunner"):
#         """Checks that the current stage has correct criterion.
#
#         Args:
#             runner: current runner
#         """
#         criterion = get_attr(
#             runner, key="criterion", inner_key=self.criterion_key
#         )
#         assert criterion is not None
#         self._criterion = criterion
#
#
# __all__ = [
#     "CriterionCallback",
# ]

from catalyst.core.callback import Callback, CallbackNode, CallbackOrder
from catalyst.core.runner import IRunner
from catalyst.metrics.misc import AverageMetric


class ICriterionCallback(Callback):
    """Criterion callback interface, abstraction over scheduler step."""

    pass


class CriterionCallback(ICriterionCallback):
    def __init__(
        self, metric_key: str = None, output_key: str = None, target_key: str = None,
    ):
        super().__init__(order=CallbackOrder.metric, node=CallbackNode.all)
        self.metric_key = metric_key
        self.output_key = output_key
        self.target_key = target_key
        self.average_metric = AverageMetric()

    def on_loader_start(self, runner: "IRunner") -> None:
        self.average_metric.reset()

    def on_batch_end(self, runner: "IRunner"):
        outputs, targets = (
            runner.batch[self.output_key],
            runner.batch[self.target_key],
        )
        outputs, targets = (
            runner.engine.sync_tensor(outputs),
            runner.engine.sync_tensor(targets),
        )

        loss = runner.criterion(outputs, targets)
        runner.batch_metrics.update({self.metric_key: loss})
        self.average_metric.update(loss.item(), len(targets))

    def on_loader_end(self, runner: "IRunner") -> None:
        loss_mean, loss_std = self.average_metric.compute()
        runner.loader_metrics.update(
            {self.metric_key: loss_mean, f"{self.metric_key}/std": loss_std}
        )
