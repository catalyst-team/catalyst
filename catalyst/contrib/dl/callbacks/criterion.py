from typing import Dict, List, Union

import deprecation

from catalyst import __version__
from catalyst.dl import MetricAggregationCallback


@deprecation.deprecated(
    deprecated_in="20.03",
    removed_in="20.04",
    current_version=__version__,
    details="Use MetricAggregationCallback instead.",
)
class CriterionAggregatorCallback(MetricAggregationCallback):
    """@TODO: Docs. Contribution is welcome."""

    def __init__(
        self,
        prefix: str,
        loss_keys: Union[str, List[str], Dict[str, float]] = None,
        loss_aggregate_fn: str = "sum",
        multiplier: float = 1.0,
    ):
        """
        Args:
            @TODO: Docs. Contribution is welcome.
        """
        super().__init__(
            prefix=prefix,
            metrics=loss_keys,
            mode=loss_aggregate_fn,
            multiplier=multiplier,
        )


__all__ = ["CriterionAggregatorCallback"]
