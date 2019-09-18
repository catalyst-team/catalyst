import torch
from torch.optim.optimizer import Optimizer


class QHAdamW(Optimizer):
    def __init__(
        self,
        params,
        lr=1e-3,
        betas=(0.995, 0.999),
        nus=(0.7, 1.0),
        weight_decay=0.0,
        eps=1e-8
    ):
        r"""
        Combines the weight decay decoupling from AdamW (Decoupled Weight
        Decay Regularization. Loshchilov and Hutter, 2019) with QHAdam
        (Quasi-hyperbolic momentum and Adam for deep learning. Ma and
        Yarats, 2019).

        https://github.com/iprally/qhadamw-pytorch/blob/master/qhadamw.py

        Args:
            params (iterable):
                iterable of parameters to optimize or dicts defining parameter
                groups
            lr (float, optional): learning rate (:math:`\alpha` from the paper)
                (default: 1e-3)
            betas (Tuple[float, float], optional): coefficients used for
                computing running averages of the gradient and its square
                (default: (0.995, 0.999))
            nus (Tuple[float, float], optional): immediate discount factors
                used to estimate the gradient and its square
                (default: (0.7, 1.0))
            eps (float, optional): term added to the denominator to improve
                numerical stability
                (default: 1e-8)
            weight_decay (float, optional): weight decay
                (L2 regularization coefficient, times two)
                (default: 0.0)
        Example:
            >>> optimizer = QHAdamW(
            ...     model.parameters(),
            ...     lr=3e-4, nus=(0.8, 1.0), betas=(0.99, 0.999))
            >>> optimizer.zero_grad()
            >>> loss_fn(model(input), target).backward()
            >>> optimizer.step()
            QHAdam paper:
        .. _`(Ma and Yarats, 2019)`: https://arxiv.org/abs/1810.06801
            AdamW paper:
        .. _`(Loshchilov and Hutter, 2019)`: https://arxiv.org/abs/1711.05101
        """
        if not 0.0 <= lr:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0.0 <= eps:
            raise ValueError(f"Invalid epsilon value: {eps}")
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 0: {betas[0]}")
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 1: {betas[1]}")
        if weight_decay < 0.0:
            raise ValueError(f"Invalid weight_decay value: {weight_decay}")

        defaults = {
            "lr": lr,
            "betas": betas,
            "nus": nus,
            "weight_decay": weight_decay,
            "eps": eps
        }
        super(QHAdamW, self).__init__(params, defaults)

    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            lr = group["lr"]
            beta1, beta2 = group["betas"]
            nu1, nu2 = group["nus"]
            weight_decay = group["weight_decay"]
            eps = group["eps"]

            for p in group["params"]:
                if p.grad is None:
                    continue

                d_p = p.grad.data
                if d_p.is_sparse:
                    raise RuntimeError(
                        "QHAdamW does not support sparse gradients")

                param_state = self.state[p]

                # Original QHAdam implementation for weight decay:
                # if weight_decay != 0:
                #    d_p.add_(weight_decay, p.data)

                d_p_sq = d_p.mul(d_p)

                if len(param_state) == 0:
                    param_state["beta1_weight"] = 0.0
                    param_state["beta2_weight"] = 0.0
                    param_state["exp_avg"] = torch.zeros_like(p.data)
                    param_state["exp_avg_sq"] = torch.zeros_like(p.data)

                param_state["beta1_weight"] = \
                    1.0 + beta1 * param_state["beta1_weight"]
                param_state["beta2_weight"] = \
                    1.0 + beta2 * param_state["beta2_weight"]

                beta1_weight = param_state["beta1_weight"]
                beta2_weight = param_state["beta2_weight"]
                exp_avg = param_state["exp_avg"]
                exp_avg_sq = param_state["exp_avg_sq"]

                beta1_adj = 1.0 - (1.0 / beta1_weight)
                beta2_adj = 1.0 - (1.0 / beta2_weight)
                exp_avg.mul_(beta1_adj).add_(1.0 - beta1_adj, d_p)
                exp_avg_sq.mul_(beta2_adj).add_(1.0 - beta2_adj, d_p_sq)

                avg_grad = exp_avg.mul(nu1)
                if nu1 != 1.0:
                    avg_grad.add_(1.0 - nu1, d_p)

                avg_grad_rms = exp_avg_sq.mul(nu2)
                if nu2 != 1.0:
                    avg_grad_rms.add_(1.0 - nu2, d_p_sq)
                avg_grad_rms.sqrt_()
                if eps != 0.0:
                    avg_grad_rms.add_(eps)

                # Original QHAdam implementation:
                # p.data.addcdiv_(-lr, avg_grad, avg_grad_rms)

                # Implementation following AdamW paper:
                p.data.add_(-weight_decay, p.data) \
                    .addcdiv_(-lr, avg_grad, avg_grad_rms)

        return loss
