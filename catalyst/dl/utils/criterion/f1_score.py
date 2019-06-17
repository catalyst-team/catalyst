import torch
from catalyst.utils import get_activation_fn


def f1_score(
    outputs: torch.Tensor,
    targets: torch.Tensor,
    beta: float = 1.0,
    eps: float = 1e-7,
    threshold: float = None,
    activation: str = "Sigmoid"
):
    """
    Source https://github.com/qubvel/segmentation_models.pytorch

    Args:
        outputs (torch.Tensor): A list of predicted elements
        targets (torch.Tensor):  A list of elements that are to be predicted
        eps (float): epsilon to avoid zero division
        beta (float): beta param for f_score
        threshold (float): threshold for outputs binarization
        activation (str): An torch.nn activation applied to the outputs.
            Must be one of ["none", "Sigmoid", "Softmax2d"]

    Returns:
        float: F_1 score
    """
    activation_fn = get_activation_fn(activation)

    outputs = activation_fn(outputs)

    if threshold is not None:
        outputs = (outputs > threshold).float()

    true_positive = torch.sum(targets * outputs)
    false_positive = torch.sum(outputs) - true_positive
    false_negative = torch.sum(targets) - true_positive

    precision_plus_recall = (1 + beta ** 2) * true_positive + \
        beta ** 2 * false_negative + false_positive + eps

    score = ((1 + beta**2) * true_positive + eps) / precision_plus_recall

    return score


__all__ = ["f1_score"]
