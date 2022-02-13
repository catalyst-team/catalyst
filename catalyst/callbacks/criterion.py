from catalyst.callbacks.metrics.functional_metric import FunctionalMetricCallback
from catalyst.core.callback import ICriterionCallback
from catalyst.core.runner import IRunner
from catalyst.utils.misc import get_attr


class CriterionCallback(FunctionalMetricCallback, ICriterionCallback):
    """Criterion callback, abstraction over criterion step.

    Args:
        input_key: input key to use for metric calculation, specifies our `y_pred`
        target_key: output key to use for metric calculation, specifies our `y_true`
        metric_key: key to store computed metric in ``runner.batch_metrics`` dictionary
        criterion_key: A key to take a criterion in case
            there are several of them, and they are in a dictionary format.

    .. note::
        Please follow the `minimal examples`_ sections for more use cases.

        .. _`minimal examples`: https://github.com/catalyst-team/catalyst#minimal-examples  # noqa: E501, W505
    """

    def __init__(
        self,
        input_key: str,
        target_key: str,
        metric_key: str,
        criterion_key: str = None,
        prefix: str = None,
        suffix: str = None,
    ):
        """Init."""
        super().__init__(
            input_key=input_key,
            target_key=target_key,
            metric_fn=self._metric_fn,
            metric_key=metric_key,
            compute_on_call=True,
            log_on_batch=True,
            prefix=prefix,
            suffix=suffix,
        )
        self.criterion_key = criterion_key
        self.criterion = None

    def _metric_fn(self, *args, **kwargs):
        return self.criterion(*args, **kwargs)

    def on_experiment_start(self, runner: "IRunner"):
        """Event handler."""
        self.criterion = get_attr(runner, key="criterion", inner_key=self.criterion_key)
        assert self.criterion is not None


__all__ = ["CriterionCallback"]
