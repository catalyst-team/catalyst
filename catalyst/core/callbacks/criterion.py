# flake8: noqa
# @TODO: code formatting issue for 20.07 release
from typing import Dict, List, Union

from catalyst.core.callbacks.metrics import IBatchMetricCallback
from catalyst.core.runner import IRunner


class CriterionCallback(IBatchMetricCallback):
    """Callback for that measures loss with specified criterion."""

    def __init__(
        self,
        input_key: Union[str, List[str], Dict[str, str]] = "targets",
        output_key: Union[str, List[str], Dict[str, str]] = "logits",
        prefix: str = "loss",
        criterion_key: str = None,
        multiplier: float = 1.0,
        **metric_kwargs,
    ):
        """
        Args:
            input_key (Union[str, List[str], Dict[str, str]]): key/list/dict
                of keys that takes values from the input dictionary
                If '__all__', the whole input will be passed to the criterion
                If None, empty dict will be passed to the criterion.
            output_key (Union[str, List[str], Dict[str, str]]): key/list/dict
                of keys that takes values from the input dictionary
                If '__all__', the whole output will be passed to the criterion
                If None, empty dict will be passed to the criterion.
            prefix (str): prefix for metrics and output key for loss
                in ``runner.batch_metrics`` dictionary
            criterion_key (str): A key to take a criterion in case
                there are several of them and they are in a dictionary format.
            multiplier (float): scale factor for the output loss.
        """
        super().__init__(
            prefix=prefix,
            input_key=input_key,
            output_key=output_key,
            multiplier=multiplier,
            **metric_kwargs,
        )
        self.criterion_key = criterion_key
        self._criterion = None

    @property
    def metric_fn(self):
        """Criterion function."""
        return self._criterion

    def on_stage_start(self, runner: IRunner):
        """Checks that the current stage has correct criterion.

        Args:
            runner: current runner
        """
        criterion = runner.get_attr(
            key="criterion", inner_key=self.criterion_key
        )
        assert criterion is not None
        self._criterion = criterion


__all__ = [
    "CriterionCallback",
]
