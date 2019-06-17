import numpy as np


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
    This function computes the average
    accuracy at k between two lists of items.

    Args:
        outputs (list): A list of predicted elements
        targets (list):  A list of elements that are to be predicted
        k (int, optional): The maximum number of predicted elements
    Returns:
        double: The average accuracy at k over the input lists
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

    Args:
        outputs (list): A list of lists of predicted elements
        targets (list): A list of lists of elements that are to be predicted
        topk (int, optional): The maximum number of predicted elements

    Returns:
        double: The mean average accuracy at k over the input lists
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


__all__ = ["accuracy", "average_accuracy", "mean_average_accuracy"]
