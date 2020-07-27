from typing import Optional, Sequence, Union

import torch


def auc(
    outputs: torch.Tensor, targets: torch.Tensor,
) -> Sequence[torch.Tensor]:

    if len(outputs) == 0:
        return 0.5

    output = None
    return output


__all__ = ["auc"]
