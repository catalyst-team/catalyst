import torch
from torch.optim.optimizer import Optimizer


def _matrix_power(matrix, power):
    # use CPU for svd for speed up
    matrix = matrix.cpu()
    u, s, v = torch.svd(matrix)
    return (u @ s.pow_(power).diag() @ v.t()).cuda()


class Shampoo(Optimizer):

    def __init__(
            self, params, lr=1e-1, momentum=0, weight_decay=0,
            epsilon=1e-4, update_freq=1):
        """

        :param params (iterable): iterable of parameters to optimize
            or dicts defining parameter groups
        :param lr (float, optional): learning rate (default: 1e-1)
        :param momentum: (float, optional): momentum factor (default: 0)
        :param weight_decay: (float, optional): weigt decay factor (default: 0)
        :param epsilon: (float, optional): momentum factor (default: 1e-4)
        :param update_freq: (int, optional):
            update frequency to compute inverse (default: 1)

        Source: https://github.com/moskomule/shampoo.pytorch
        """
        defaults = dict(
            lr=lr, momentum=momentum, weight_decay=weight_decay,
            epsilon=epsilon, update_freq=update_freq)
        super(Shampoo, self).__init__(params, defaults)

    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad.data
                order = grad.ndimension()
                original_size = grad.size()
                state = self.state[p]
                momentum = group["momentum"]
                weight_decay = group["weight_decay"]

                if len(state) == 0:
                    state["step"] = 0
                    if momentum > 0:
                        state["momentum_buffer"] = grad.clone()
                    for dim_id, dim in enumerate(grad.size()):
                        # precondition matrices
                        state[f"precond_{dim_id}"] = (
                                group["epsilon"] *
                                torch.eye(dim, out=grad.new(dim, dim)))
                        state[f"inv_precond_{dim_id}"] = \
                            grad.new(dim, dim).zero_()

                if momentum > 0:
                    # grad = (1 - moment) * grad(t) + moment * grad(t-1)
                    # and grad(-1) = grad(0)
                    grad.mul_(1 - momentum).add_(
                        momentum, state["momentum_buffer"])

                if weight_decay > 0:
                    grad.add_(group["weight_decay"], p.data)

                # See Algorithm 2 for detail
                for dim_id, dim in enumerate(grad.size()):
                    precond = state[f"precond_{dim_id}"]
                    inv_precond = state[f"inv_precond_{dim_id}"]

                    # mat_{dim_id}(grad)
                    grad = grad.transpose_(0, dim_id).contiguous()
                    transposed_size = grad.size()
                    grad = grad.view(dim, -1)

                    grad_t = grad.t()
                    precond.add_(grad @ grad_t)
                    if state["step"] % group["update_freq"] == 0:
                        inv_precond.copy_(_matrix_power(precond, -1 / order))

                    if dim_id == order - 1:
                        # finally
                        grad = grad_t @ inv_precond
                        # grad: (-1, last_dim)
                        grad = grad.view(original_size)
                    else:
                        # if not final
                        grad = inv_precond @ grad
                        # grad (dim, -1)
                        grad = grad.view(transposed_size)

                state["step"] += 1
                state["momentum_buffer"] = grad
                p.data.add_(-group["lr"], grad)

        return loss
