from typing import List
from collections import OrderedDict
import glob
import logging
import os
from pathlib import Path

import torch

logger = logging.getLogger(__name__)


def average_weights(state_dicts: List[dict]) -> OrderedDict:
    """
    Averaging of input weights.

    Args:
        state_dicts (List[dict]): Weights to average

    Returns:
        Averaged weights
    """
    # source https://gist.github.com/qubvel/70c3d5e4cddcde731408f478e12ef87b

    average_dict = OrderedDict()
    for k in state_dicts[0].keys():
        average_dict[k] = torch.true_divide(
            sum(state_dict[k] for state_dict in state_dicts), len(state_dicts),
        )
    return average_dict


def load_weight(path: str) -> dict:
    """
    Load weights of a model.

    Args:
        path (str): Path to model weights

    Returns:
        Weights
    """
    weights = torch.load(path)
    if "model_state_dict" in weights:
        weights = weights["model_state_dict"]
    return weights


def generate_averaged_weights(logdir: Path, models_mask: str) -> OrderedDict:
    """
    Averaging of input weights and saving them.

    Args:
        logdir (Path): Path to logs directory
        models_mask (str): globe-like pattern for models to average

    Returns:
        Averaged weights
    """
    models_pathes = glob.glob(os.path.join(logdir, "checkpoints", models_mask))

    all_weights = [load_weight(path) for path in models_pathes]
    averaged_dict = average_weights(all_weights)

    return averaged_dict
