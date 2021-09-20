# flake8: noqa
from typing import Optional

import numpy as np

import torch
from torch import nn
from torch.autograd import Function, Variable


class Pointwise(nn.Module):
    """Base class for pointwise loss functions.

    Pointwise approaches look at a single document at a time in the loss function.
    For a single documents predict it relevance to the query in time.
    The score is independent for the order of the documents that are in the query's results.

    Input space: single document d1
    Output space: scores or relevant classes
    """

    def forward(self, score: torch.Tensor):
        raise NotImplementedError()


class PairwiseLoss(nn.Module):
    """Base class for pairwise loss functions.

    Pairwise approached looks at a pair of documents at a time in the loss function.
    Given a pair of documents the algorithm try to come up with the optimal ordering
    For that pair and compare it with the ground truth. The goal for the ranker is to
    minimize the number of inversions in ranker.

    Input space: pairs of documents (d1, d2)
    Output space: preferences (yes/no) for a given doc. pair
    """

    @staticmethod
    def _assert_equal_size(positive_score: torch.Tensor, negative_score: torch.Tensor) -> None:
        if positive_score.size() != negative_score.size():
            raise ValueError(f"Shape mismatch: {positive_score.size()}, {negative_score.size()}")

    def forward(self, positive_score: torch.Tensor, negative_score: torch.Tensor):
        raise NotImplementedError()


class ListwiseLoss(nn.Module):
    """Base class for listwise loss functions.

    Listwise approach directly looks at the entire list of documents and comes up with
    an optimal ordering for it.

    Input space: document set
    Output space: permutations - ranking of documents
    """

    @staticmethod
    def _assert_equal_size(outputs: torch.Tensor, targets: torch.Tensor) -> None:
        if outputs.size() != targets.size():
            raise ValueError(f"Shape mismatch: {outputs.size()}, {targets.size()}")

    def forward(self, outputs: torch.Tensor, targets: torch.Tensor):
        raise NotImplementedError()


class BPRLoss(PairwiseLoss):
    """Bayesian Personalised Ranking loss function.

    It has been proposed in `BPRLoss\: Bayesian Personalized Ranking from Implicit Feedback`_.

    .. _BPRLoss\: Bayesian Personalized Ranking from Implicit Feedback:
        https://arxiv.org/pdf/1205.2618.pdf

    Args:
        gamma (float): Small value to avoid division by zero. Default: ``1e-10``.

    Example:

    .. code-block:: python

        import torch
        from catalyst.contrib.nn.criterion import recsys

        pos_score = torch.randn(3, requires_grad=True)
        neg_score = torch.randn(3, requires_grad=True)

        output = recsys.BPRLoss()(pos_score, neg_score)
        output.backward()
    """

    def __init__(self, gamma=1e-10) -> None:
        super().__init__()
        self.gamma = gamma

    def forward(self, positive_score: torch.Tensor, negative_score: torch.Tensor) -> torch.Tensor:
        """Forward propagation method for the BPR loss.

        Args:
            positive_score: Tensor containing predictions for known positive items.
            negative_score: Tensor containing predictions for sampled negative items.

        Returns:
            computed loss
        """
        self._assert_equal_size(positive_score, negative_score)

        loss = -torch.log(self.gamma + torch.sigmoid(positive_score - negative_score))
        return loss.mean()


class LogisticLoss(PairwiseLoss):
    """Logistic loss function.

    Example:

    .. code-block:: python

        import torch
        from catalyst.contrib.nn.criterion import recsys

        pos_score = torch.randn(3, requires_grad=True)
        neg_score = torch.randn(3, requires_grad=True)

        output = recsys.LogisticLoss()(pos_score, neg_score)
        output.backward()
    """

    def __init__(self) -> None:
        super().__init__()

    def forward(self, positive_score: torch.Tensor, negative_score: torch.Tensor) -> torch.Tensor:
        """Forward propagation method for the logistic loss.

        Args:
            positive_score: Tensor containing predictions for known positive items.
            negative_score: Tensor containing predictions for sampled negative items.

        Returns:
            computed loss
        """
        self._assert_equal_size(positive_score, negative_score)

        positives_loss = 1.0 - torch.sigmoid(positive_score)
        negatives_loss = torch.sigmoid(negative_score)

        loss = positives_loss + negatives_loss

        return loss.mean()


class HingeLoss(PairwiseLoss):
    """Hinge loss function.

    Example:

    .. code-block:: python

        import torch
        from catalyst.contrib.nn.criterion import recsys

        pos_score = torch.randn(3, requires_grad=True)
        neg_score = torch.randn(3, requires_grad=True)

        output = recsys.HingeLoss()(pos_score, neg_score)
        output.backward()
    """

    def __init__(self) -> None:
        super().__init__()

    def forward(self, positive_score: torch.Tensor, negative_score: torch.Tensor) -> torch.Tensor:
        """Forward propagation method for the hinge loss.

        Args:
            positive_score: Tensor containing predictions for known positive items.
            negative_score: Tensor containing predictions for sampled negative items.

        Returns:
            computed loss
        """
        self._assert_equal_size(positive_score, negative_score)

        loss = torch.clamp(1.0 - (positive_score - negative_score), min=0.0)
        return loss.mean()


class AdaptiveHingeLoss(PairwiseLoss):
    """Adaptive hinge loss function.

    Takes a set of predictions for implicitly negative items, and selects those
    that are highest, thus sampling those negatives that are closes to violating
    the ranking implicit in the pattern of user interactions.

    Example:

    .. code-block:: python

        import torch
        from catalyst.contrib.nn.criterion import recsys

        pos_score = torch.randn(3, requires_grad=True)
        neg_scores = torch.randn(5, 3, requires_grad=True)

        output = recsys.AdaptiveHingeLoss()(pos_score, neg_scores)
        output.backward()
    """

    def __init__(self) -> None:
        super().__init__()
        self._hingeloss = HingeLoss()

    def forward(self, positive_score: torch.Tensor, negative_scores: torch.Tensor) -> torch.Tensor:
        """Forward propagation method for the adaptive hinge loss.

        Args:
            positive_score: Tensor containing predictions for known positive items.
            negative_scores: Iterable of tensors containing predictions for sampled negative items.
                More tensors increase the likelihood of finding ranking-violating pairs,
                but risk overfitting.

        Returns:
            computed loss
        """
        self._assert_equal_size(positive_score, negative_scores[0])

        highest_negative_score, _ = torch.max(negative_scores, 0)

        return self._hingeloss.forward(positive_score, highest_negative_score.squeeze())


class WARP(Function):
    """Autograd function of WARP loss."""

    @staticmethod
    def forward(
        ctx: nn.Module,
        outputs: torch.Tensor,
        targets: torch.Tensor,
        max_num_trials: Optional[int] = None,
    ):

        batch_size = targets.size()[0]
        if max_num_trials is None:
            max_num_trials = targets.size()[1] - 1

        positive_indices = torch.zeros(outputs.size())
        negative_indices = torch.zeros(outputs.size())
        L = torch.zeros(outputs.size()[0])

        all_labels_idx = torch.arange(targets.size()[1])

        Y = float(targets.size()[1])
        J = torch.nonzero(targets)

        for i in range(batch_size):

            msk = torch.ones(targets.size()[1], dtype=bool)

            # Find the positive label for this example
            j = J[i, 1]
            positive_indices[i, j] = 1
            msk[j] = False

            # initialize the sample_score_margin
            sample_score_margin = -1
            num_trials = 0

            neg_labels_idx = all_labels_idx[msk]

            while (sample_score_margin < 0) and (num_trials < max_num_trials):  # type: ignore

                # randomly sample a negative label, example from here:
                # https://github.com/pytorch/pytorch/issues/16897
                neg_idx = neg_labels_idx[torch.randint(0, neg_labels_idx.size(0), (1,))]
                msk[neg_idx] = False
                neg_labels_idx = all_labels_idx[msk]

                num_trials += 1
                # calculate the score margin
                sample_score_margin = 1 + outputs[i, neg_idx] - outputs[i, j]

            if sample_score_margin < 0:
                # checks if no violating examples have been found
                continue
            else:
                loss_weight = np.log(np.floor((Y - 1) / (num_trials)))
                L[i] = loss_weight
                negative_indices[i, neg_idx] = 1  # type: ignore

        loss = L * (
            1
            - torch.sum(positive_indices * outputs, dim=1)
            + torch.sum(negative_indices * outputs, dim=1)
        )

        ctx.save_for_backward(outputs, targets)
        ctx.L = L
        ctx.positive_indices = positive_indices
        ctx.negative_indices = negative_indices

        return torch.sum(loss, dim=0, keepdim=True)

    # This function has only a single output, so it gets only one gradient
    @staticmethod
    def backward(ctx, grad_output):
        outputs, targets = ctx.saved_variables
        L = Variable(torch.unsqueeze(ctx.L, 1), requires_grad=False)

        positive_indices = Variable(ctx.positive_indices, requires_grad=False)
        negative_indices = Variable(ctx.negative_indices, requires_grad=False)
        grad_input = grad_output * L * (negative_indices - positive_indices)

        return grad_input, None, None


class WARPLoss(ListwiseLoss):
    """Weighted Approximate-Rank Pairwise (WARP) loss function.

    It has been proposed in `WSABIE\: Scaling Up To Large Vocabulary Image Annotation`_ paper.

    .. _WSABIE\: Scaling Up To Large Vocabulary Image Annotation:
        https://static.googleusercontent.com/media/research.google.com/en//pubs/archive/37180.pdf

    WARP loss randomly sample output labels of a model, until it finds a pair
    which it knows are wrongly labelled and will then only apply an update to
    these two incorrectly labelled examples.

    Adapted from:
    https://github.com/gabrieltseng/datascience-projects/blob/master/misc/warp.py

    Args:
        max_num_trials: Number of attempts allowed to find a violating negative example.
            In practice it means that we optimize for ranks 1 to max_num_trials-1.

    Example:

    .. code-block:: python

        import torch
        from catalyst.contrib.nn.criterion import recsys

        outputs = torch.randn(5, 3, requires_grad=True)
        targets = torch.randn(5, 3, requires_grad=True)

        output = recsys.WARPLoss()(outputs, targets)
        output.backward()
    """

    def __init__(self, max_num_trials: Optional[int] = None):
        super().__init__()
        self.max_num_trials = max_num_trials

    def forward(self, outputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Forward propagation method for the WARP loss.

        Args:
            outputs: Iterable of tensors containing predictions for all items.
            targets: Iterable of tensors containing true labels for all items.

        Returns:
            computed loss
        """
        self._assert_equal_size(outputs, targets)
        return WARP.apply(outputs, targets, self.max_num_trials)


class RocStarLoss(PairwiseLoss):
    """Roc-star loss function.

    Smooth approximation for ROC-AUC. It has been proposed in
    `Roc-star\: An objective function for ROC-AUC that actually works`_.

    .. _Roc-star\: An objective function for ROC-AUC that actually works:
        https://github.com/iridiumblue/roc-star

    Adapted from:
    https://github.com/iridiumblue/roc-star/issues/2

    Args:
        delta: Param from the article. Default: ``1.0``.
        sample_size: Number of examples to take for ROC AUC approximation. Default: ``100``.
        sample_size_gamma: Number of examples to take for Gamma parameter approximation.
            Default: ``1000``.
        update_gamma_each: Number of steps after which to recompute gamma value.
            Default: ``50``.

    Example:

        .. code-block:: python

            import torch
            from catalyst.contrib.nn.criterion import recsys

            outputs = torch.randn(5, 1, requires_grad=True)
            targets = torch.randn(5, 1, requires_grad=True)

            output = recsys.RocStarLoss()(outputs, targets)
            output.backward()
    """

    def __init__(
        self,
        delta: float = 1.0,
        sample_size: int = 100,
        sample_size_gamma: int = 1000,
        update_gamma_each: int = 50,
    ):
        super().__init__()
        self.delta = delta
        self.sample_size = sample_size
        self.sample_size_gamma = sample_size_gamma
        self.update_gamma_each = update_gamma_each
        self.steps = 0
        self.gamma = None
        size = max(sample_size, sample_size_gamma)

        # Randomly init labels
        self.outputs_history = torch.rand((size + 2, 1))
        self.targets_history = torch.cat(
            (torch.randint(2, (size, 1)), torch.LongTensor([[0], [1]]))
        )

    def forward(self, outputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Forward propagation method for the roc-star loss.

        Args:
            outputs: Tensor of model predictions in [0, 1] range. Shape ``(B x 1)``.
            targets: Tensor of true labels in {0, 1}. Shape ``(B x 1)``.

        Returns:
            computed loss
        """
        self._assert_equal_size(outputs, targets)

        if torch.sum(targets) == 0 or torch.sum(targets) == targets.shape[0]:
            return torch.sum(outputs) * 1e-8

        if self.steps % self.update_gamma_each == 0:
            self._update_gamma()
        self.steps += 1

        positive = outputs[targets > 0]
        negative = outputs[targets < 1]

        # Take last `sample_size` elements from history
        outputs_history = self.outputs_history[-self.sample_size :]
        targets_history = self.targets_history[-self.sample_size :]

        positive_history = outputs_history[targets_history > 0]
        negative_history = outputs_history[targets_history < 1]

        if positive.size(0) > 0:
            diff = negative_history.view(1, -1) + self.gamma - positive.view(-1, 1)
            loss_positive = nn.functional.relu(diff ** 2).mean()
        else:
            loss_positive = 0

        if negative.size(0) > 0:
            diff = negative.view(1, -1) + self.gamma - positive_history.view(-1, 1)
            loss_negative = nn.functional.relu(diff ** 2).mean()
        else:
            loss_negative = 0

        loss = loss_negative + loss_positive

        # Update FIFO queue
        batch_size = outputs.size(0)
        self.outputs_history = torch.cat(
            (self.outputs_history[batch_size:], outputs.clone().detach())
        )
        self.targets_history = torch.cat(
            (self.targets_history[batch_size:], targets.clone().detach())
        )

        return loss

    def _update_gamma(self):
        # Take last `sample_size_gamma` elements from history
        outputs = self.outputs_history[-self.sample_size_gamma :]
        targets = self.targets_history[-self.sample_size_gamma :]

        positive = outputs[targets > 0]
        negative = outputs[targets < 1]

        # Create matrix of size sample_size_gamma x sample_size_gamma
        diff = positive.view(-1, 1) - negative.view(1, -1)
        AUC = (diff > 0).type(torch.float).mean()
        num_wrong_ordered = (1 - AUC) * diff.flatten().size(0)

        # Adjunct gamma, so that among correct ordered samples `delta * num_wrong_ordered`
        # were considered ordered incorrectly with gamma added
        correct_ordered = diff[diff > 0].flatten().sort().values
        idx = min(int(num_wrong_ordered * self.delta), len(correct_ordered) - 1)
        if idx >= 0:
            self.gamma = correct_ordered[idx]


__all__ = [
    "AdaptiveHingeLoss",
    "BPRLoss",
    "HingeLoss",
    "LogisticLoss",
    "RocStarLoss",
    "WARPLoss",
]
