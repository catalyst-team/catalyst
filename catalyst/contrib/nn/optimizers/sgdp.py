"""
AdamP
Copyright (c) 2020-present NAVER Corp.
MIT license

Original source code: https://github.com/clovaai/AdamP
"""

import math

import torch
import torch.nn.functional as F
from torch.optim.optimizer import Optimizer, required


class SGDP(Optimizer):
    """Implements SGDP algorithm.

    The SGDP variant was proposed in
    `Slowing Down the Weight Norm Increase in Momentum-based Optimizers`_.

    Args:
        params: iterable of parameters to optimize or dicts defining
            parameter groups
        lr: learning rate
        momentum (float, optional): momentum factor (default: 0)
        weight_decay (float, optional): weight decay (L2 penalty) (default: 0)
        dampening (float, optional): dampening for momentum (default: 0)
        nesterov (bool, optional): enables Nesterov momentum (default: False)
        eps (float, optional): term added to the denominator to improve
            numerical stability (default: 1e-8)
        delta: threshold that determines whether
            a set of parameters is scale invariant or not (default: 0.1)
        wd_ratio: relative weight decay applied on scale-invariant
            parameters compared to that applied on scale-variant parameters
            (default: 0.1)

    .. _Slowing Down the Weight Norm Increase in Momentum-based Optimizers:
        https://arxiv.org/abs/2006.08217
    """

    def __init__(
        self,
        params,
        lr=required,
        momentum=0,
        weight_decay=0,
        dampening=0,
        nesterov=False,
        eps=1e-8,
        delta=0.1,
        wd_ratio=0.1,
    ):
        """

        Args:
            params: iterable of parameters to optimize
                or dicts defining parameter groups
            lr: learning rate
            momentum (float, optional): momentum factor (default: 0)
            weight_decay (float, optional): weight decay (L2 penalty)
                (default: 0)
            dampening (float, optional): dampening for momentum (default: 0)
            nesterov (bool, optional): enables Nesterov momentum
                (default: False)
            eps (float, optional): term added to the denominator to improve
                numerical stability (default: 1e-8)
            delta: threshold that determines whether
                a set of parameters is scale invariant or not (default: 0.1)
            wd_ratio: relative weight decay applied on scale-invariant
                parameters compared to that applied on scale-variant parameters
                (default: 0.1)
        """
        defaults = dict(  # noqa: C408
            lr=lr,
            momentum=momentum,
            dampening=dampening,
            weight_decay=weight_decay,
            nesterov=nesterov,
            eps=eps,
            delta=delta,
            wd_ratio=wd_ratio,
        )
        super(SGDP, self).__init__(params, defaults)

    def _channel_view(self, x):
        return x.view(x.size(0), -1)

    def _layer_view(self, x):
        return x.view(1, -1)

    def _cosine_similarity(self, x, y, eps, view_func):
        x = view_func(x)
        y = view_func(y)

        return F.cosine_similarity(x, y, dim=1, eps=eps).abs_()

    def _projection(self, p, grad, perturb, delta, wd_ratio, eps):
        wd = 1
        expand_size = [-1] + [1] * (len(p.shape) - 1)
        for view_func in [self._channel_view, self._layer_view]:

            cosine_sim = self._cosine_similarity(grad, p.data, eps, view_func)

            if cosine_sim.max() < delta / math.sqrt(view_func(p.data).size(1)):
                p_n = p.data / view_func(p.data).norm(dim=1).view(expand_size).add_(eps)
                perturb -= p_n * view_func(p_n * perturb).sum(dim=1).view(expand_size)
                wd = wd_ratio

                return perturb, wd

        return perturb, wd

    def step(self, closure=None):
        """
        Performs a single optimization step (parameter update).

        Arguments:
            closure: A closure that reevaluates the model and
                returns the loss. Optional for most optimizers.

        Returns:
            computed loss
        """
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            momentum = group["momentum"]
            dampening = group["dampening"]
            nesterov = group["nesterov"]

            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad.data
                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state["momentum"] = torch.zeros_like(p.data)

                # SGD
                buf = state["momentum"]
                buf.mul_(momentum).add_(grad, alpha=1 - dampening)
                if nesterov:
                    d_p = grad + momentum * buf
                else:
                    d_p = buf

                # Projection
                wd_ratio = 1
                if len(p.shape) > 1:
                    d_p, wd_ratio = self._projection(
                        p, grad, d_p, group["delta"], group["wd_ratio"], group["eps"],
                    )

                # Weight decay
                if group["weight_decay"] > 0:
                    p.data.mul_(
                        1 - group["lr"] * group["weight_decay"] * wd_ratio / (1 - momentum)
                    )

                # Step
                p.data.add_(d_p, alpha=-group["lr"])

        return loss


__all__ = ["SGDP"]
