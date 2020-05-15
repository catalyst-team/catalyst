from operator import itemgetter

from catalyst.data.sampler import BalancecBatchSampler


def test():
    # TODO

    labels = [0, 1, 3, 2, 1, 3, 1, 0, 0, 2, 3]

    sampler = BalancecBatchSampler(labels, p=3, k=3)
    print(itemgetter(*sampler)(labels))
