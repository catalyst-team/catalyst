from typing import List, Union
from collections import OrderedDict
import glob
import os
from pathlib import Path

import torch


def _load_weights(path: str) -> dict:
    """
    Load weights of a model.

    Args:
        path: Path to model weights

    Returns:
        Weights
    """
    weights = torch.load(path, map_location=lambda storage, loc: storage)
    if "model_state_dict" in weights:
        weights = weights["model_state_dict"]
    return weights


def average_weights(state_dicts: List[dict]) -> OrderedDict:
    """
    Averaging of input weights.

    Args:
        state_dicts: Weights to average

    Raises:
        KeyError: If states do not match

    Returns:
        Averaged weights
    """
    # source https://gist.github.com/qubvel/70c3d5e4cddcde731408f478e12ef87b
    params_keys = None
    for i, state_dict in enumerate(state_dicts):
        model_params_keys = list(state_dict.keys())
        if params_keys is None:
            params_keys = model_params_keys
        elif params_keys != model_params_keys:
            raise KeyError(
                "For checkpoint {}, expected list of params: {}, "
                "but found: {}".format(i, params_keys, model_params_keys)
            )

    average_dict = OrderedDict()
    for k in state_dicts[0].keys():
        average_dict[k] = torch.div(
            sum(state_dict[k] for state_dict in state_dicts), len(state_dicts),
        )

    return average_dict


def get_averaged_weights_by_path_mask(
    path_mask: str, logdir: Union[str, Path] = None,
) -> OrderedDict:
    """
    Averaging of input weights and saving them.

    Args:
        path_mask: globe-like pattern for models to average
        logdir: Path to logs directory

    Returns:
        Averaged weights
    """
    if logdir is None:
        models_pathes = glob.glob(path_mask)
    else:
        models_pathes = glob.glob(os.path.join(logdir, "checkpoints", path_mask))

    all_weights = [_load_weights(path) for path in models_pathes]
    averaged_dict = average_weights(all_weights)

    return averaged_dict


__all__ = ["average_weights", "get_averaged_weights_by_path_mask"]
