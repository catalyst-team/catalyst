import numpy as np
import torch


def precision(outputs, targets, topk=(1,)):
    """
    Computes the precision@k for the specified values of k
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


def average_precision(outputs, targets, k=10):
    """
    Computes the average precision at k.
    This function computes the average precision at k between two lists of
    items.
    Parameters
    ----------
    outputs : list
                A list of predicted elements (order does matter)
    targets : list
             A list of elements that are to be predicted (order doesn't matter)
    k : int, optional
        The maximum number of predicted elements
    Returns
    -------
    score : double
            The average precision at k over the input lists
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


def mean_average_precision(outputs, targets, topk=(1,)):
    """
    Computes the mean average precision at k.
    This function computes the mean average precision at k between two lists
    of lists of items.
    Parameters
    ----------
    outputs : list
                A list of lists of predicted elements
                (order matters in the lists)
    targets : list
             A list of lists of elements that are to be predicted
             (order doesn't matter in the lists)
    topk : int, optional
        The maximum number of predicted elements
    Returns
    -------
    score : double
            The mean average precision at k over the input lists
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
        ap = np.mean([average_precision(p, a, k) for a, p in zip(targets, pred)])
        res.append(ap)
    return res


def dice(
        outputs,
        targets,
        eps: float = 1e-7,
        activation: str = 'sigmoid'):
    """
    Computes the average precision at k.
    This function computes the average precision at k between two lists of
    items.
    Parameters
    ----------
    outputs : list
                A list of predicted elements (order does matter)
    targets : list
             A list of elements that are to be predicted (order doesn't matter)
    eps : float
        TODO
    activation: str
        TODO
    Returns
    -------
    score : double
            Dice score
    """
    if activation is None or activation == "none":
        activation_fn = lambda x: x
    elif activation == "sigmoid":
        activation_fn = torch.nn.Sigmoid()
    elif activation == "softmax":
        activation_fn = torch.nn.Softmax2d()
    else:
        raise NotImplementedError("Dice is only implemented for sigmoid and softmax")

    outputs = activation_fn(outputs)
    intersection = torch.sum(targets * outputs)
    sum_ = torch.sum(targets) + torch.sum(outputs) + eps

    return (2 * intersection + eps) / sum_


def jaccard(outputs, targets, eps: float = 1e-7):
    """
    Computes the average precision at k.
    This function computes the average precision at k between two lists of
    items.
    Parameters
    ----------
    outputs : list
                A list of predicted elements (order does matter)
    targets : list
             A list of elements that are to be predicted (order doesn't matter)
    eps : float
    Returns
    -------
    score : double
            Jaccard score
    """
    intersection = torch.sum(targets * outputs)
    union = torch.sum(targets) + torch.sum(outputs) - intersection + eps
    return (intersection + eps) / union
