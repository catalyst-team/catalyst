# flake8: noqa
import torch
from torch import nn


class Pointwise(nn.Module):
    """
    Pointwise approaches look at a single document at a time in the loss function. 
    For a single documetns predict it relevance to the query in time. 
    The score is independeent for the order of the docuemtns that are in the query's resluts

    Input space: single document d1
    Output space: scores or relevant classes
    """
    def __init__(self) -> None:
        super().__init__()
    
    def forward(self, score:torch.Tensor):
        pass

class PairwiseLoss(nn.Module):
    """
    Pairwise approached looks at a pair ofd documents at a time in the loss function.
    Given a pair of documents the algorithm try to come up with the optimal ordering 
    For that pair and compare it with the ground truth. The goal for the ranker is to 
    minimize the number of inversions in ranker. 

    Input space: pairs of documents (d1, d2)
    Output space: preferences (yes/no) for a given doc.pair
    """
    def __init__(self) -> None:
        super().__init__()
    
    def forward(self, positive_score:torch.Tensor, negative_score:torch.Tensor):
        pass

class ListWiseLoss(nn.Module):
    """
    Listwise approach directly looks at the entire list of documents and comes up with an
    optimal ordering for it. 

    Input space: document set
    Output space: permutations - ranking of documents 
    """
    def __init__(self) -> None:
        super().__init__()
    
    def forward(self, score_list:torch.Tensor):
        pass


class BPRLoss(PairwiseLoss):
    """"
    BPRLoss, based on Bayesian Personalized Ranking
    https://arxiv.org/pdf/1205.2618

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
    
    def forward(self, positive_score:torch.Tensor, negative_score:torch.Tensor)->torch.Tensor:
        """
        Args
            positive_predictions: torch.LongTensor
                Tensor containing predictions for known positive items.
            negative_predictions: torch.LongTensor
                Tensor containing predictions for sampled negative items.
        """
        loss = -torch.log(self.gamma + torch.sigmoid(positive_score - negative_score))
        return loss.mean()


class WARPLoss(PairwiseLoss):
    """
    Weighted Approximate-Rank Pairwise (WARP) loss function for implicit feedback. 
    https://static.googleusercontent.com/media/research.google.com/en//pubs/archive/37180.pdf

     WARP loss randomly sample output labels of a model, until it finds a pair which it knows are wrongly labelled 
     and will then only apply an update to these two incorrectly labelled examples.
    """
    def __init__(self) -> None:
        super().__init__()
    
    def forward(self, positive_score:torch.Tensor, negative_score:torch.Tensor)->torch.Tensor:
        """
        Args
            positive_predictions: torch.LongTensor
                Tensor containing predictions for known positive items.
            negative_predictions: torch.LongTensor
                Tensor containing predictions for sampled negative items.
        """
        
        highest_negative_score, _ = torch.max(negative_score, 0).squezee()

        loss = torch.clamp(highest_negative_score -
                       positive_score +
                       1.0, 0.0)

        return loss.mean()


class LogisticLoss(Pointwise):
    """
    Logistic loss
    """

    def __init__(self, ) -> None:
        super().__init__()
    
    def forward(self, positive_score, negative_score):
        """
        Args
            positive_predictions: torch.LongTensor
                Tensor containing predictions for known positive items.
            negative_predictions: torch.LongTensor
                Tensor containing predictions for sampled negative items.
        """
        
        positives_loss = (1.0 - torch.sigmoid(positive_score))
        negatives_loss = torch.sigmoid(negative_score)
        
        loss = (positives_loss + negatives_loss)

        return loss.mean()
