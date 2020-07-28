from typing import List, Tuple
from abc import ABC, abstractmethod
from collections import Counter
from itertools import combinations, product
from random import sample
from sys import maxsize

import numpy as np

import torch
from torch import Tensor

from catalyst.contrib.utils.misc import find_value_ids
from catalyst.utils.torch import normalize

# order in the triplets: (anchor, positive, negative)
TTriplets = Tuple[Tensor, Tensor, Tensor]
TTripletsIds = Tuple[List[int], List[int], List[int]]


class InBatchTripletsSampler(ABC):
    """
    Base class for a triplets samplers.
    We expect that the child instances of this class
    will be used to forming triplets inside the batches.
    The batches must contain at least 2 samples for
    each class and at least 2 different classes,
    such behaviour can be garantee via using
    catalyst.data.sampler.BalanceBatchSampler

    But you are not limited to using it in any other way.
    """

    @staticmethod
    def _check_input_labels(labels: List[int]) -> None:
        """
        The input must satisfy the conditions described in
        the class documentation.

        Args:
            labels: labels of the samples in the batch
        """
        labels_counter = Counter(labels)
        assert all(n > 1 for n in labels_counter.values())
        assert len(labels_counter) > 1

    @abstractmethod
    def _sample(self, features: Tensor, labels: List[int]) -> TTripletsIds:
        """
        This method includes the logic of sampling/selecting triplets
        inside the batch. It can be based on information about
        the distance between the features, or the
        choice can be made randomly.

        Args:
            features: has the shape of [batch_size, feature_size]
            labels: labels of the samples in the batch

        Returns: indeces of the batch samples to forming triplets.
        """
        raise NotImplementedError

    def sample(self, features: Tensor, labels: List[int]) -> TTriplets:
        """
        Args:
            features: has the shape of [batch_size, feature_size]
            labels: labels of the samples in the batch

        Returns:
            the batch of the triplets in the order below:
            (anchor, positive, negative)
        """
        self._check_input_labels(labels=labels)

        ids_anchor, ids_pos, ids_neg = self._sample(
            features=features, labels=labels
        )

        return features[ids_anchor], features[ids_pos], features[ids_neg]


class AllTripletsSampler(InBatchTripletsSampler):
    """
    This sampler selects all the possible triplets for the given labels
    """

    def __init__(self, max_output_triplets: int = maxsize):
        """
        Args:
            max_output_triplets: with the strategy of choosing all
                the triplets, their number in the batch can be very large,
                because of it we can sample only random part of them,
                determined by this parameter.
        """
        self._max_out_triplets = max_output_triplets

    def _sample(self, *_: Tensor, labels: List[int]) -> TTripletsIds:
        """
        Args:
            labels: labels of the samples in the batch
            *_: note, that we ignore features argument

        Returns: indeces of triplets
        """
        num_labels = len(labels)

        triplets = []
        for label in set(labels):
            ids_pos_cur = set(find_value_ids(labels, label))
            ids_neg_cur = set(range(num_labels)) - ids_pos_cur

            pos_pairs = list(combinations(ids_pos_cur, r=2))

            tri = [(a, p, n) for (a, p), n in product(pos_pairs, ids_neg_cur)]
            triplets.extend(tri)

        triplets = sample(triplets, min(len(triplets), self._max_out_triplets))
        ids_anchor, ids_pos, ids_neg = zip(*triplets)

        return list(ids_anchor), list(ids_pos), list(ids_neg)


class HardTripletsSampler(InBatchTripletsSampler):
    """
    This sampler selects hardest triplets based on distances between features:
    the hardest positive sample has the maximal distance to the anchor sample,
    the hardest negative sample has the minimal distance to the anchor sample.

    """

    def __init__(self, need_norm: bool = False):
        """
        Args:
            need_norm: set True if features normalisation is needed
        """
        self._need_norm = need_norm

    def _sample(self, features: Tensor, labels: List[int]) -> TTripletsIds:
        """
        This method samples the hardest triplets inside the batch.

        Args:
            features: has the shape of [batch_size, feature_size]
            labels: labels of the samples in the batch

        Returns:
            the batch of the triplets in the order below:
            (anchor, positive, negative)
        """
        assert features.shape[0] == len(labels)

        if self._need_norm:
            features = normalize(samples=features)

        dist_mat = torch.cdist(x1=features, x2=features, p=2)

        ids_anchor, ids_pos, ids_neg = self._sample_from_distmat(
            distmat=dist_mat, labels=labels
        )

        return ids_anchor, ids_pos, ids_neg

    @staticmethod
    def _sample_from_distmat(
        distmat: Tensor, labels: List[int]
    ) -> TTripletsIds:
        """
        This method samples the hardest triplets based on the given
        distances matrix. It chooses each sample in the batch as an
        anchor and then finds the harderst positive and negative pair.

        Args:
            distmat: matrix of distances between the features
            labels: labels of the samples in the batch

        Returns:
            the batch of triplets in the order below:
            (anchor, positive, negative)
        """
        ids_all = set(range(len(labels)))

        ids_anchor, ids_pos, ids_neg = [], [], []

        for i_anch, label in enumerate(labels):
            ids_label = set(find_value_ids(it=labels, value=label))

            ids_pos_cur = np.array(list(ids_label - {i_anch}), int)
            ids_neg_cur = np.array(list(ids_all - ids_label), int)

            i_pos = ids_pos_cur[distmat[i_anch, ids_pos_cur].argmax()]
            i_neg = ids_neg_cur[distmat[i_anch, ids_neg_cur].argmin()]

            ids_anchor.append(i_anch)
            ids_pos.append(i_pos)
            ids_neg.append(i_neg)

        return ids_anchor, ids_pos, ids_neg
