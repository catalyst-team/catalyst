import torch
import numpy as np
from sklearn.metrics import f1_score, mean_absolute_error


def precision(output, target, topk=(1,)):
    """
    Computes the precision@k for the specified values of k
    """
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


def get_iou_vector(A, B):
    batch_size = A.shape[0]
    B = np.where(B > 0.5, 1, 0)
    metric = []
    for batch in range(batch_size):
        t, p = A[batch], B[batch]
        if np.count_nonzero(t) == 0 and np.count_nonzero(p) > 0:
            metric.append(0)
            continue
        if np.count_nonzero(t) >= 1 and np.count_nonzero(p) == 0:
            metric.append(0)
            continue
        if np.count_nonzero(t) == 0 and np.count_nonzero(p) == 0:
            metric.append(1)
            continue

        intersection = np.logical_and(t, p)
        union = np.logical_or(t, p)
        iou = np.sum(intersection > 0) / np.sum(union > 0)
        thresholds = np.arange(0.5, 1, 0.05)
        s = []
        for thresh in thresholds:
            s.append(iou > thresh)
        metric.append(np.mean(s))

    return np.mean(metric)


def apk(actual, predicted, k=10):
    """
    Computes the average precision at k.
    This function computes the average prescision at k between two lists of
    items.
    Parameters
    ----------
    actual : list
             A list of elements that are to be predicted (order doesn't matter)
    predicted : list
                A list of predicted elements (order does matter)
    k : int, optional
        The maximum number of predicted elements
    Returns
    -------
    score : double
            The average precision at k over the input lists
    """
    if len(predicted)>k:
        predicted = predicted[:k]

    score = 0.0
    num_hits = 0.0

    for i,p in enumerate(predicted):
        if p in actual and p not in predicted[:i]:
            num_hits += 1.0
            score += num_hits / (i+1.0)

    if not actual:
        return 0.0

    return score / min(len(actual), k)


def mapk(predicted, actual, topk=(1,)):
    """
    Computes the mean average precision at k.
    This function computes the mean average prescision at k between two lists
    of lists of items.
    Parameters
    ----------
    actual : list
             A list of lists of elements that are to be predicted
             (order doesn't matter in the lists)
    predicted : list
                A list of lists of predicted elements
                (order matters in the lists)
    k : int, optional
        The maximum number of predicted elements
    Returns
    -------
    score : double
            The mean average precision at k over the input lists
    """
    maxk = max(topk)
    _, pred = predicted.topk(maxk, 1, True, True)

    actual = actual.data.cpu().numpy().tolist()
    actual_list = []
    for a in actual:
        actual_list.append([a])
    actual = actual_list
    pred = pred.tolist()

    res = []
    for k in topk:
        ap = np.mean([apk(a,p,k) for a,p in zip(actual, pred)])
        res.append(ap)
    return res


def dice_accuracy(prob, truth, threshold=0.5,  is_average=True):
    batch_size = prob.size(0)
    p = prob.detach().view(batch_size,-1)
    t = truth.detach().view(batch_size,-1)

    p = p>threshold
    t = t>0.5
    correct = ( p == t).float()
    accuracy = correct.sum(1)/p.size(1)

    if is_average:
        accuracy = accuracy.sum()/batch_size
        return accuracy
    else:
        return accuracy


def F_score(label, output, start=0.2, end=0.5, step=0.01):
    return max([f1_score(label, (output > th), average='macro')
                for th in np.arange(start, end, step)])


def mae(label, output):
    return mean_absolute_error(label, output)


if __name__ == '__main__':
    import torch
    actual = torch.from_numpy(np.array([0, 0]))
    predict = torch.from_numpy(np.array([[0.5, 0.1, 0.1, 0.1, 0.2], [0.2, 0.5, 0.1, 0.3, 0.0]]))
    print(actual.shape)
    print(predict.shape)
    m = mapk(predict, actual, topk=(1, 5, ))
    print(m)