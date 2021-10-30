from typing import Sequence

import torch


def r2_squared(outputs: torch.Tensor, targets: torch.Tensor) -> Sequence[torch.Tensor]:
    """
    Computes regression r2 squared.

    Args:
        outputs: model outputs
            with shape [bs; 1]
        targets: ground truth
            with shape [bs; 1]

    Returns:
        float of computed r2 squared

    Examples:

    .. code-block:: python

        import torch
        from catalyst import metrics
        metrics.r2_squared(
            outputs=torch.tensor([0, 1, 2]),
            targets=torch.tensor([0, 1, 2]),
        )
        # tensor([1.])


    .. code-block:: python

        import torch
        from catalyst import metrics
        metrics.r2_squared(
            outputs=torch.tensor([2.5, 0.0, 2, 8]),
            targets=torch.tensor([3, -0.5, 2, 7]),
        )
        # tensor([0.9486])
    """
    total_sum_of_squares = torch.sum(
        torch.pow(targets.float() - torch.mean(targets.float()), 2)
    ).view(-1)
    residual_sum_of_squares = torch.sum(torch.pow(targets.float() - outputs.float(), 2)).view(-1)
    output = 1 - residual_sum_of_squares / total_sum_of_squares
    return output


__all__ = ["r2_squared"]
