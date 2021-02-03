# from typing import List
#
# from catalyst.callbacks.metric import BatchMetricCallback
# from catalyst.metrics.functional import wrap_class_metric2dict, wrap_metric_fn_with_activation
# from catalyst.metrics.recall import recall
#
#
# class RecallCallback(BatchMetricCallback):
#     """Recall score metric callback."""
#
#     def __init__(
#         self,
#         input_key: str = "targets",
#         output_key: str = "logits",
#         prefix: str = "recall",
#         activation: str = "Softmax",
#         per_class: bool = False,
#         class_args: List[str] = None,
#         **kwargs,
#     ):
#         """
#         Args:
#             input_key: input key to use for iou calculation
#                 specifies our ``y_true``
#             output_key: output key to use for iou calculation;
#                 specifies our ``y_pred``
#             prefix: key for the metric's name
#             activation: An torch.nn activation applied to the outputs.
#                 Must be one of ``'none'``, ``'Sigmoid'``, or ``'Softmax'``
#             per_class: boolean flag to log per class metrics,
#                 or use mean/macro statistics otherwise
#             class_args: class names to display in the logs.
#                 If None, defaults to indices for each class, starting from 0
#             **kwargs: key-value params to pass to the metric
#
#         .. note::
#             For ``**kwargs`` info, please follow
#             ``catalyst.callbacks.metric.BatchMetricCallback`` and
#             ``catalyst.metrics.recall.recall`` docs
#         """
#         metric_fn = wrap_metric_fn_with_activation(metric_fn=recall, activation=activation)
#         metric_fn = wrap_class_metric2dict(metric_fn, per_class=per_class, class_args=class_args)
#         super().__init__(
#             prefix=prefix,
#             metric_fn=metric_fn,
#             input_key=input_key,
#             output_key=output_key,
#             **kwargs,
#         )
#
#
# __all__ = ["RecallCallback"]
