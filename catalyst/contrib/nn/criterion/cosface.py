import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.loss import _WeightedLoss


class CosFaceLoss(_WeightedLoss):
    """Implementation of CosFace loss for metric learning.

    .. _CosFace: Large Margin Cosine Loss for Deep Face Recognition
        https://arxiv.org/abs/1801.09414
    """

    def __init__(
        self,
        embedding_size: int,
        num_classes: int,
        s: float = 64.0,
        m: float = 0.35,
        weight: torch.Tensor = None,
        size_average=None,
        ignore_index: int = -100,
        reduce=None,
        reduction: str = "mean",
    ):
        """
        Args:
            embedding_size (int): size of each input sample.
            num_classes (int): size of each output sample.
            s (float): norm of input feature,
                Default: ``64.0``.
            m (float): margin.
                Default: ``0.35``.
            weight (float, optional): – a manual rescaling weight given to each class.
                If given, has to be a Tensor of size `num_classes`.
                Default: ``None``.
            size_average (bool, optional): Deprecated (see :attr:`reduction`). By default,
                the losses are averaged over each loss element in the batch. Note that for
                some losses, there are multiple elements per sample. If the field :attr:`size_average`
                is set to ``False``, the losses are instead summed for each minibatch. Ignored
                when reduce is ``False``.
                Default: ``True``
            reduce (bool, optional): Deprecated (see :attr:`reduction`). By default, the
                losses are averaged or summed over observations for each minibatch depending
                on :attr:`size_average`. When :attr:`reduce` is ``False``, returns a loss per
                batch element instead and ignores :attr:`size_average`.
                Default: ``True``
            reduction (string, optional): Specifies the reduction to apply to the output:
                ``'none'`` | ``'mean'`` | ``'sum'``. ``'none'``: no reduction will be applied,
                ``'mean'``: the sum of the output will be divided by the number of
                elements in the output, ``'sum'``: the output will be summed. Note: :attr:`size_average`
                and :attr:`reduce` are in the process of being deprecated, and in the meantime,
                specifying either of those two args will override :attr:`reduction`.
                Default: ``'mean'``
        """
        super(CosFaceLoss, self).__init__(
            weight, size_average, reduce, reduction
        )
        self.ignore_index = ignore_index
        self.embedding_size = embedding_size
        self.num_classes = num_classes
        self.s = s
        self.m = m

        self.projection = nn.Parameter(
            torch.FloatTensor(num_classes, embedding_size)
        )
        nn.init.xavier_uniform_(self.projection)

    def forward(
        self, input: torch.Tensor, target: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            input (torch.Tensor): input features,
                expected shapes BxF.
            target (torch.Tensor): target classes,
                expected shapes B.

        Returns:
            torch.Tensor with loss value.
        """
        cosine = F.linear(F.normalize(input), F.normalize(self.projection))
        phi = cosine - self.m
        one_hot = torch.zeros(cosine.size()).to(input.device)
        one_hot.scatter_(1, target.view(-1, 1).long(), 1)
        logits = (one_hot * phi) + ((1.0 - one_hot) * cosine)
        logits *= self.s

        return F.cross_entropy(
            logits,
            target,
            weight=self.weight,
            ignore_index=self.ignore_index,
            reduction=self.reduction,
        )
