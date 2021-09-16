from typing import List

import numpy as np

import torch


def mixup_batch(
    batch: List[torch.Tensor], alpha: float = 0.2, mode: str = "replace"
) -> List[torch.Tensor]:
    """

    Args:
        batch: batch to which you want to apply augmentation
        alpha: beta distribution a=b parameters. Must be >=0. The closer alpha to zero the
            less effect of the mixup.
        mode: algorithm used for muxup: ``"replace"`` | ``"add"``. If "replace"
            then replaces the batch with a mixed one, while the batch size is not changed
            If "add", concatenates mixed examples to the current ones, the batch size increases
            by 2 times.

    Returns:
        augmented batch

    """
    assert alpha >= 0, "alpha must be>=0"
    assert mode in ("add", "replace"), f"mode must be in 'add', 'replace', get: {mode}"

    batch_size = batch[0].shape[0]
    beta = np.random.beta(alpha, alpha, batch_size).astype(np.float32)
    indexes = np.arange(batch_size)
    # index shift by 1
    indexes_2 = (indexes + 1) % batch_size
    for idx, targets in enumerate(batch):
        device = targets.device
        targets_shape = [batch_size] + [1] * len(targets.shape[1:])
        key_beta = torch.as_tensor(beta.reshape(targets_shape), device=device)
        targets = targets * key_beta + targets[indexes_2] * (1 - key_beta)

        if mode == "replace":
            batch[idx] = targets
        else:
            # mode == 'add'
            batch[idx] = torch.cat([batch[idx], targets])
    return batch
