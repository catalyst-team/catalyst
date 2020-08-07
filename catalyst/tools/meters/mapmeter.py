"""
The mAP meter measures the mean average precision over all classes.
"""
from typing import Optional

import torch

from catalyst.tools.meters import APMeter, meter


class mAPMeter(meter.Meter):  # noqa: N801
    """
    This meter is a wrapper for
    :py:class:`catalyst.tools.meters.apmeter.APMeter`.
    The mAPMeter is designed to operate on `NxK` Tensors `output` and
    `target`, and optionally a `Nx1` Tensor weight where:

    1. The `output` contains model output scores for `N` examples and `K`
    classes that ought to be higher when the model is more convinced that
    the example should be positively labeled, and smaller when the model
    believes the example should be negatively labeled
    (for instance, the output of a sigmoid function)

    2. The `target` contains only values 0 (for negative examples) and 1
    (for positive examples)

    3. The `weight` ( > 0) represents weight
    for each sample.
    """

    def __init__(self):
        """Constructor method for the ``mAPMeter`` class."""
        super(mAPMeter, self).__init__()
        self.apmeter = APMeter()

    def reset(self) -> None:
        """Reset `self.apmeter`."""
        self.apmeter.reset()

    def add(
        self,
        output: torch.Tensor,
        target: torch.Tensor,
        weight: Optional[torch.Tensor] = None,
    ) -> None:
        """Update `self.apmeter`.

        Args:
            output (Tensor): Model output scores as `NxK` tensor
            target (Tensor): Target scores as `NxK` tensor
            weight (Tensor, optional): Weight values for each sample
                as `Nx1` Tensor
        """
        self.apmeter.add(output, target, weight)

    def value(self):
        """Returns mean of `self.apmeter` value.

        Returns:
            torch.Tensor: mAP scalar tensor
        """
        return self.apmeter.value().mean()


__all__ = ["mAPMeter"]
