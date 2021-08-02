# flake8: noqa
from typing import Optional

import numpy as np
import torch
from torch import nn
from torch.autograd import Function, Variable


class Pointwise(nn.Module):
    """
    Pointwise approaches look at a single document at a time in the loss function. 
    For a single documents predict it relevance to the query in time.
    The score is independent for the order of the documents that are in the query's results.

    Input space: single document d1
    Output space: scores or relevant classes
    """

    def __init__(self) -> None:
        super().__init__()

    def forward(self, score: torch.Tensor):
        raise NotImplementedError()


class PairwiseLoss(nn.Module):
    """
    Pairwise approached looks at a pair of documents at a time in the loss function.
    Given a pair of documents the algorithm try to come up with the optimal ordering 
    For that pair and compare it with the ground truth. The goal for the ranker is to 
    minimize the number of inversions in ranker. 

    Input space: pairs of documents (d1, d2)
    Output space: preferences (yes/no) for a given doc.pair
    """

    @staticmethod
    def _assert_equal_size(positive_score: torch.Tensor, negative_score: torch.Tensor) -> None:
        if positive_score.size() != negative_score.size():
            raise ValueError(f"Shape mismatch: {positive_score.size()}, {negative_score.size()}")

    def __init__(self) -> None:
        super().__init__()

    def forward(self, positive_score: torch.Tensor, negative_score: torch.Tensor):
        raise NotImplementedError()


class ListWiseLoss(nn.Module):
    """
    Listwise approach directly looks at the entire list of documents and comes up with an
    optimal ordering for it. 

    Input space: document set
    Output space: permutations - ranking of documents 
    """

    @staticmethod
    def _assert_equal_size(input_: torch.Tensor, target: torch.Tensor) -> None:
        if input_.size() != target.size():
            raise ValueError(f"Shape mismatch: {input_.size()}, {target.size()}")

    def __init__(self) -> None:
        super().__init__()

    def forward(self, input_: torch.Tensor, target: torch.Tensor):
        raise NotImplementedError()


class BPRLoss(PairwiseLoss):
    """ Implementation of 
    `BPRLoss, based on Bayesian Personalized Ranking`_ paper.


    .. _BPRLoss: Bayesian Personalized Ranking from Implicit Feedback:
        https://arxiv.org/pdf/1205.2618.pdf

    Args:
        - gamma(float): Small value to avoid division by zero
    
    Example:
    .. code-block:: python
        import torch
        from torch.contrib import recsys
        
        pos_score = torch.randn(3, requires_grad=True)
        neg_score = torch.randn(3, requires_grad=True)
        
        output = recsys.BPRLoss()(pos_score, neg_score)
        output.backward()
    """

    def __init__(self, gamma=1e-10) -> None:
        super().__init__()
        self.gamma = gamma

    def forward(self, positive_score: torch.Tensor, negative_score: torch.Tensor) -> torch.Tensor:
        """
        Args
            positive_predictions: torch.Tensor
                Tensor containing predictions for known positive items.
            negative_predictions: torch.Tensor
                Tensor containing predictions for sampled negative items.
        """
        self._assert_equal_size(positive_score, negative_score)

        loss = -torch.log(self.gamma + torch.sigmoid(positive_score - negative_score))
        return loss.mean()


class LogisticLoss(PairwiseLoss):
    """
    Logistic loss
    """

    def __init__(self,) -> None:
        super().__init__()

    def forward(self, positive_score: torch.Tensor, negative_score: torch.Tensor) -> torch.Tensor:
        """
        Args
            positive_predictions: torch.Tensor
                Tensor containing predictions for known positive items.
            negative_predictions: torch.Tensor
                Tensor containing predictions for sampled negative items.
        """
        self._assert_equal_size(positive_score, negative_score)

        positives_loss = 1.0 - torch.sigmoid(positive_score)
        negatives_loss = torch.sigmoid(negative_score)

        loss = positives_loss + negatives_loss

        return loss.mean()


class HingeLoss(PairwiseLoss):
    """
    Hinge loss
    """

    def __init__(self,) -> None:
        super().__init__()

    def forward(self, positive_score: torch.Tensor, negative_score: torch.Tensor) -> torch.Tensor:
        """
        Args
            positive_predictions: torch.Tensor
                Tensor containing predictions for known positive items.
            negative_predictions: torch.Tensor
                Tensor containing predictions for sampled negative items.
        """
        self._assert_equal_size(positive_score, negative_score)

        loss = torch.clamp(1.0 - (positive_score - negative_score), min=0.0)
        return loss.mean()


class AdaptiveHingeLoss(PairwiseLoss):
    """
    Adaptive hinge pairwise loss function. Takes a set of predictions
    for implicitly negative items, and selects those that are highest,
    thus sampling those negatives that are closes to violating the
    ranking implicit in the pattern of user interactions.
    """

    def __init__(self) -> None:
        super().__init__()

    def forward(self, positive_score: torch.Tensor, negative_scores: torch.Tensor) -> torch.Tensor:
        """
        Args
            positive_score: torch.Tensor
                Tensor containing predictions for known positive items.
            negative_scores: torch.Tensor
                Iterable of tensors containing predictions for sampled negative items.
                More tensors increase the likelihood of finding ranking-violating
                pairs, but risk overfitting.
        """
        self._assert_equal_size(positive_score, negative_scores[0])

        highest_negative_score, _ = torch.max(negative_scores, 0)

        return HingeLoss().forward(positive_score, highest_negative_score.squeeze())


class WARP(Function):
    """
    Autograd function of WARP loss
    """

    @staticmethod
    def forward(
        ctx: nn.Module,
        input_: torch.Tensor,
        target: torch.Tensor,
        max_num_trials: Optional[int] = None,
    ):

        batch_size = target.size()[0]
        if max_num_trials is None:
            max_num_trials = target.size()[1] - 1

        positive_indices = torch.zeros(input_.size())
        negative_indices = torch.zeros(input_.size())
        L = torch.zeros(input_.size()[0])

        all_labels_idx = torch.arange(target.size()[1])

        Y = float(target.size()[1])
        J = torch.nonzero(target)

        for i in range(batch_size):

            msk = torch.ones(target.size()[1], dtype=bool)

            # Find the positive label for this example
            j = J[i, 1]
            positive_indices[i, j] = 1
            msk[j] = False

            # initialize the sample_score_margin
            sample_score_margin = -1
            num_trials = 0

            neg_labels_idx = all_labels_idx[msk]

            while (sample_score_margin < 0) and (num_trials < max_num_trials):  # type: ignore

                # randomly sample a negative label, example from here: https://github.com/pytorch/pytorch/issues/16897
                neg_idx = neg_labels_idx[torch.randint(0, neg_labels_idx.size(0), (1,))]
                msk[neg_idx] = False
                neg_labels_idx = all_labels_idx[msk]

                num_trials += 1
                # calculate the score margin
                sample_score_margin = 1 + input_[i, neg_idx] - input_[i, j]

            if sample_score_margin < 0:
                # checks if no violating examples have been found
                continue
            else:
                loss_weight = np.log(np.floor((Y - 1) / (num_trials)))
                L[i] = loss_weight
                negative_indices[i, neg_idx] = 1  # type: ignore

        loss = L * (
            1
            - torch.sum(positive_indices * input_, dim=1)
            + torch.sum(negative_indices * input_, dim=1)
        )

        ctx.save_for_backward(input_, target)
        ctx.L = L
        ctx.positive_indices = positive_indices
        ctx.negative_indices = negative_indices

        return torch.sum(loss, dim=0, keepdim=True)

    # This function has only a single output, so it gets only one gradient
    @staticmethod
    def backward(ctx, grad_output):
        input_, target = ctx.saved_variables
        L = Variable(torch.unsqueeze(ctx.L, 1), requires_grad=False)

        positive_indices = Variable(ctx.positive_indices, requires_grad=False)
        negative_indices = Variable(ctx.negative_indices, requires_grad=False)
        grad_input = grad_output * L * (negative_indices - positive_indices)

        return grad_input, None, None


class WARPLoss(ListWiseLoss):
    """ Implementation of 
    Weighted Approximate-Rank Pairwise (WARP) loss function for implicit feedback,
    based on paper `WSABIE: Scaling Up To Large Vocabulary Image Annotation`_
    
    .. _WSABIE: Scaling Up To Large Vocabulary Image Annotation:
        https://static.googleusercontent.com/media/research.google.com/en//pubs/archive/37180.pdf
    
    WARP loss randomly sample output labels of a model, until it finds a pair which it knows are wrongly labelled 
     and will then only apply an update to these two incorrectly labelled examples.
    """

    def __init__(self, max_num_trials: Optional[int] = None):
        super(WARPLoss, self).__init__()
        self.max_num_trials = max_num_trials

    def forward(self, input_: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        self._assert_equal_size(input_, target)
        return WARP.apply(input_, target, self.max_num_trials)
