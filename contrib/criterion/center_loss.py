import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.autograd.function import Function


class CenterLoss(nn.Module):
    def __init__(self, num_classes, feature_dim):
        super().__init__()
        self.centers = nn.Parameter(torch.randn(num_classes, feature_dim))
        self.loss_fn = CenterLossFunc.apply
        self.feature_dim = feature_dim

    def forward(self, feature, label):
        batch_size = feature.size(0)
        feature = feature.view(batch_size, 1, 1, -1).squeeze()
        # To check the dim of centers and features
        if feature.size(1) != self.feature_dim:
            raise ValueError(
                "Center\"s dim: {0} "
                "should be equal to input feature\"s dim: {1}".format(
                    self.feature_dim, feature.size(1)
                )
            )
        return self.loss_fn(feature, label, self.centers)


class CenterLossFunc(Function):
    @staticmethod
    def forward(ctx, feature, label, centers):
        ctx.save_for_backward(feature, label, centers)
        centers_batch = centers.index_select(0, label.long())
        return (feature - centers_batch).pow(2).sum(1).sum(0) / 2.0

    @staticmethod
    def backward(ctx, grad_output):
        feature, label, centers = ctx.saved_tensors
        centers_batch = centers.index_select(0, label.long())
        diff = centers_batch - feature
        # init every iteration
        counts = centers.new(centers.size(0)).fill_(1)
        ones = centers.new(label.size(0)).fill_(1)
        grad_centers = centers.new(centers.size()).fill_(0)
        counts = counts.scatter_add_(0, label.long(), ones)
        # print counts, grad_centers
        grad_centers.scatter_add_(
            0,
            label.unsqueeze(1).expand(feature.size()).long(), diff
        )

        grad_centers = grad_centers / counts.view(-1, 1)

        return Variable(-grad_output.data * diff), None, Variable(grad_centers)
