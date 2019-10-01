import numpy as np
from catalyst.utils import get_activation_fn


def accuracy(
        outputs,
        targets,
        topk=(1,),
        threshold: float = None,
        activation: str = None
):
    """
    Computes the accuracy.

    It can be used either for:
        - multi-class task:
            -you can use topk.
            -threshold and activation are not required.
            -targets is a tensor: batch_size
            -outputs is a tensor: batch_size x num_classes
            -computes the accuracy@k for the specified values of k.
        - OR multi-label task, in this case:
            -you must specify threshold and activation
            -topk will not be used
            (because of there is no method to apply top-k in
            multi-label classification).
            -outputs, targets are tensors with shape: batch_size x num_classes
            -targets is a tensor with binary vectors
    """
    activation_fn = get_activation_fn(activation)
    outputs = activation_fn(outputs)

    if threshold:
        outputs = (outputs > threshold).long()

    # multi-label classification
    if len(targets.shape) > 1 and targets.size(1) > 1:
        res = (targets.long() == outputs.long()).sum().float() / np.prod(
            targets.shape)
        return [res]

    max_k = max(topk)
    batch_size = targets.size(0)

    if len(outputs.shape) == 1 or outputs.shape[1] == 1:
        pred = outputs.t()
    else:
        _, pred = outputs.topk(max_k, 1, True, True)
        pred = pred.t()
    correct = pred.eq(targets.long().view(1, -1).expand_as(pred))

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


def mean_average_accuracy(outputs, targets, topk=(1,)):
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
