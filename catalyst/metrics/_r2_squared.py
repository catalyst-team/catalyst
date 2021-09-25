from typing import Optional

import torch

from catalyst.metrics._metric import ICallbackLoaderMetric


class R2Squared(ICallbackLoaderMetric):
    """This metric accumulates r2 score along loader

    Args:
        compute_on_call: if True, allows compute metric's value on call
        prefix: metric prefix
        suffix: metric suffix
    """

    def __init__(
        self,
        compute_on_call: bool = True,
        prefix: Optional[str] = None,
        suffix: Optional[str] = None,
    ) -> None:
        """Init R2Squared"""
        super().__init__(compute_on_call=compute_on_call, prefix=prefix, suffix=suffix)
        self.metric_name = f"{self.prefix}r2squared{self.suffix}"
        self.num_examples = 0
        self.delta_sum = 0
        self.y_sum = 0
        self.y_sq_sum = 0

    def reset(self, num_batches: int, num_samples: int) -> None:
        """
        Reset metrics fields
        """
        self.num_examples = 0
        self.delta_sum = 0
        self.y_sum = 0
        self.y_sq_sum = 0

    def update(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> None:
        """
        Update accumulated data with new batch
        """
        self.num_examples += len(y_true)
        self.delta_sum += torch.sum(torch.pow(y_pred - y_true, 2))
        self.y_sum += torch.sum(y_true)
        self.y_sq_sum += torch.sum(torch.pow(y_true, 2))

    def compute(self) -> torch.Tensor:
        """
        Return accumulated metric
        """
        return 1 - self.delta_sum / (self.y_sq_sum - (self.y_sum ** 2) / self.num_examples)

    def compute_key_value(self) -> torch.Tensor:
        """
        Return key-value
        """
        r2squared = self.compute()
        output = {self.metric_name: r2squared}
        return output


__all__ = ["R2Squared"]
