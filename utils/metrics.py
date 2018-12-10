import torch
import numpy as np
from sklearn.metrics import f1_score, mean_absolute_error, precision_score, recall_score
from sklearn.metrics import average_precision_score, precision_recall_curve
from sklearn.metrics.classification import precision_recall_fscore_support


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


def jaccard(y_true, y_pred):
    y_true = y_true.detach().cpu().numpy()
    y_pred = y_pred.detach().cpu().numpy()
    intersection = (y_true * y_pred).sum()
    union = y_true.sum() + y_pred.sum() - intersection
    return (intersection + 1e-15) / (union + 1e-15)


def dice(y_true, y_pred):
    return (2 * (y_true * y_pred).sum() + 1e-15) / (y_true.sum() + y_pred.sum() + 1e-15)


def F1(label, output, th=0.2, start=0.2, end=0.5, step=0.01):
    precision, recall, f_score, true_sum = precision_recall_fscore_support(label, output > th, beta=1, average="macro")
    return precision, recall, f_score, th


def fbeta(y_pred, y_true, thresh:float=0.2, beta:float=2, eps:float=1e-9, sigmoid:bool=True):
    "Computes the f_beta between `y_pred` and `y_true` in a multi-classification task."
    beta2 = beta**2
    if sigmoid: y_pred = y_pred.sigmoid()
    y_pred = (y_pred>thresh).float()
    y_true = y_true.float()
    TP = (y_pred*y_true).sum(dim=1)
    prec = TP/(y_pred.sum(dim=1)+eps)
    rec = TP/(y_true.sum(dim=1)+eps)
    res = (prec*rec)/(prec*beta2+rec+eps)*(1+beta2)
    return res.mean()


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