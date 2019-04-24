import numpy as np
import torch

from .utils import get_activation_by_name


def accuracy(outputs, targets, topk=(1, )):
    """
    Computes the accuracy@k for the specified values of k
    """
    max_k = max(topk)
    batch_size = targets.size(0)

    _, pred = outputs.topk(max_k, 1, True, True)
    pred = pred.t()
    correct = pred.eq(targets.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


def average_accuracy(outputs, targets, k=10):
    """
    Computes the average accuracy at k.
    This function computes the average accuracy at k
        between two lists of items.
    Parameters
    ----------
    outputs : list
        A list of predicted elements
    targets : list
        A list of elements that are to be predicted
    k : int, optional
        The maximum number of predicted elements
    Returns
    -------
    score : double
            The average accuracy at k over the input lists
    """
    if len(outputs) > k:
        outputs = outputs[:k]

    score = 0.0
    num_hits = 0.0

    for i, predict in enumerate(outputs):
        if predict in targets and predict not in outputs[:i]:
            num_hits += 1.0
            score += num_hits / (i + 1.0)

    if not targets:
        return 0.0

    return score / min(len(targets), k)


def mean_average_accuracy(outputs, targets, topk=(1, )):
    """
    Computes the mean average accuracy at k.
    This function computes the mean average accuracy at k between two lists
        of lists of items.
    Parameters
    ----------
    outputs : list
                A list of lists of predicted elements
    targets : list
             A list of lists of elements that are to be predicted
    topk : int, optional
        The maximum number of predicted elements
    Returns
    -------
    score : double
            The mean average accuracy at k over the input lists
    """
    max_k = max(topk)
    _, pred = outputs.topk(max_k, 1, True, True)

    targets = targets.data.cpu().numpy().tolist()
    actual_list = []
    for a in targets:
        actual_list.append([a])
    targets = actual_list
    pred = pred.tolist()

    res = []
    for k in topk:
        ap = np.mean(
            [average_accuracy(p, a, k) for a, p in zip(targets, pred)]
        )
        res.append(ap)
    return res


def dice(outputs, targets, eps: float = 1e-7, activation: str = "sigmoid"):
    """
    Computes the dice metric
        Args:
        outputs (list):  A list of predicted elements
        targets (list): A list of elements that are to be predicted
        eps (float): epsilon
        activation (str): An torch.nn activation applied to the outputs.
            Must be one of ['none', 'sigmoid', 'softmax2d']
    Returns:
        double:  Dice score
    """
    activation_fn = get_activation_by_name(activation)

    outputs = activation_fn(outputs)
    intersection = torch.sum(targets * outputs)
    sum_ = torch.sum(targets) + torch.sum(outputs) + eps

    return (2 * intersection + eps) / sum_


def iou(
    outputs: torch.Tensor,
    targets: torch.Tensor,
    eps: float = 1e-7,
    threshold: float = 0.5,
    activation: str = "sigmoid"
):
    """
    Args:
        outputs (torch.Tensor): A list of predicted elements
        targets (torch.Tensor):  A list of elements that are to be predicted
        eps (float): epsilon to avoid zero division
        threshold (float): threshold for outputs binarization
        activation (str): An torch.nn activation applied to the outputs.
            Must be one of ['none', 'sigmoid', 'softmax2d']
    Returns:
        float: IoU (Jaccard) score
    """
    activation_fn = get_activation_by_name(activation)

    outputs = activation_fn(outputs)

    if threshold is not None:
        outputs = (outputs > threshold).float()

    intersection = torch.sum(targets * outputs)
    union = torch.sum(targets) + torch.sum(outputs) - intersection + eps

    return (intersection + eps) / union


jaccard = iou


def soft_iou(
    outputs: torch.Tensor,
    targets: torch.Tensor,
    eps: float = 1e-7,
    threshold: float = 0.5,
    activation: str = "sigmoid"
):
    """
    Args:
        outputs (torch.Tensor): A list of predicted elements
        targets (torch.Tensor):  A list of elements that are to be predicted
        eps (float): epsilon to avoid zero division
        threshold (float): threshold for outputs binarization
        activation (str): An torch.nn activation applied to the outputs.
            Must be one of ['none', 'sigmoid', 'softmax2d']
    Returns:
        float: SoftIoU (SoftJaccard) score
    """
    jaccards = []
    for class_i in range(outputs.shape[1]):
        jaccard_i = iou(
            outputs[:, class_i, :, :],
            targets[:, class_i, :, :],
            eps=eps,
            threshold=threshold,
            activation=activation,
        )
        jaccards.append(jaccard_i)
    return torch.mean(torch.stack(jaccards))


def f_score(
    outputs: torch.Tensor,
    targets: torch.Tensor,
    beta: float = 1,
    eps: float = 1e-7,
    threshold: float = 0.5,
    activation: str = "sigmoid"
):
    """
    Source:
        https://github.com/qubvel/segmentation_models.pytorch
    Args:
        outputs (torch.Tensor): A list of predicted elements
        targets (torch.Tensor):  A list of elements that are to be predicted
        eps (float): epsilon to avoid zero division
        beta (float): beta param for f_score
        threshold (float): threshold for outputs binarization
        activation (str): An torch.nn activation applied to the outputs.
            Must be one of ['none', 'sigmoid', 'softmax2d']
    Returns:
        float: F_1 score
    """
    activation_fn = get_activation_by_name(activation)

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
