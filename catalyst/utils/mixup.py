from typing import Dict, List

import numpy as np
import torch


def mixup_batch(
    batch: Dict[str, torch.Tensor], keys: List[str], alpha: float = 0.2, mode: str = "replace"
) -> Dict[str, torch.Tensor]:
    """

    Args:
        batch: batch to which you want to apply augmentation
        keys: batch keys to which you want to apply augmentation
        alpha: beta distribution a=b parameters. Must be >=0. The closer alpha to zero the
            less effect of the mixup.
        mode: mode determines the method of use. Must be in ["replace", "add"]. If "replace"
            then replaces the batch with a mixed one, while the batch size is not changed
            If "add", concatenates mixed examples to the current ones, the batch size increases
            by 2 times.

    Returns:
        augmented batch

    """
    assert isinstance(keys, list), f"keys must be list[str], get: {type(keys)}"
    assert alpha >= 0, "alpha must be>=0"
    assert mode in ["add", "replace"], f"mode must be in 'add', 'replace', get: {mode}"

    batch_size = batch[keys[0]].shape[0]
    beta = np.random.beta(alpha, alpha, batch_size).astype(np.float32)
    indexes = np.array(list(range(batch_size)))
    # index shift by 1
    indexes_2 = (indexes + 1) % batch_size
    for key in keys:
        targets = batch[key]
        device = targets.device
        targets_shape = [batch_size] + [1] * len(targets.shape[1:])
        key_beta = torch.Tensor(beta.reshape(targets_shape)).to(device)
        targets = targets * key_beta + targets[indexes_2] * (1 - key_beta)

        if mode == "replace":
            batch[key] = targets
        else:
            # mode == 'add'
            batch[key] = torch.cat([batch[key], targets])
    return batch
