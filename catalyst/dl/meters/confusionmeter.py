import numpy as np

from . import meter


class ConfusionMeter(meter.Meter):
    """Maintains a confusion matrix for a given calssification problem.

    The ConfusionMeter constructs a confusion matrix for a multi-class
    classification problems. It does not support multi-label, multi-class
    problems: for such problems, please use MultiLabelConfusionMeter.

    Args:
        k (int): number of classes in the classification problem
        normalized (boolean): Determines whether or not the confusion matrix
            is normalized or not

    """

    def __init__(self, k, normalized=False):
        super(ConfusionMeter, self).__init__()
        self.conf = np.ndarray((k, k), dtype=np.int32)
        self.normalized = normalized
        self.k = k
        self.reset()

    def reset(self):
        self.conf.fill(0)

    def add(self, predicted, target):
        """Computes the confusion matrix of K x K size where K is no of classes

        Args: predicted (tensor): Can be an N x K tensor of predicted scores
        obtained from the model for N examples and K classes or an N-tensor
        of integer values between 0 and K-1. target (tensor): Can be a
        N-tensor of integer values assumed to be integer values between 0 and
        K-1 or N x K tensor, where targets are assumed to be provided as
        one-hot vectors

        """
        predicted = predicted.cpu().numpy()
        target = target.cpu().numpy()

        assert predicted.shape[0] == target.shape[0], \
            "number of targets and predicted outputs do not match"

        if np.ndim(predicted) != 1:
            assert predicted.shape[1] == self.k, \
                "number of predictions does not match size of confusion matrix"
            predicted = np.argmax(predicted, 1)
        else:
            assert (predicted.max() < self.k) and (predicted.min() >= 0), \
                "predicted values are not between 1 and k"

        onehot_target = np.ndim(target) != 1
        if onehot_target:
            assert target.shape[1] == self.k, \
                "Onehot target does not match size of confusion matrix"
            assert (target >= 0).all() and (target <= 1).all(), \
                "in one-hot encoding, target values should be 0 or 1"
            assert (target.sum(1) == 1).all(), \
                "multi-label setting is not supported"
            target = np.argmax(target, 1)
        else:
            assert (predicted.max() < self.k) and (predicted.min() >= 0), \
                "predicted values are not between 0 and k-1"

        # hack for bincounting 2 arrays together
        x = predicted + self.k * target
        bincount_2d = np.bincount(x.astype(np.int32),
                                  minlength=self.k ** 2)
        assert bincount_2d.size == self.k ** 2
        conf = bincount_2d.reshape((self.k, self.k))

        self.conf += conf

    def value(self):
        """
        Returns:
            Confustion matrix of K rows and K columns, where rows corresponds
            to ground-truth targets and columns corresponds to predicted
            targets.
        """
        if self.normalized:
            conf = self.conf.astype(np.float32)
            return conf / conf.sum(1).clip(min=1e-12)[:, None]
        else:
            return self.conf
