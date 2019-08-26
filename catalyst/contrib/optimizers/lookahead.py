from typing import Dict
import itertools

from torch.optim import Optimizer


class Lookahead(Optimizer):
    def __init__(
        self,
        base_optimizer: Optimizer,
        alpha: float = 0.5,
        k: int = 6
    ):
        """
        taken from https://github.com/lonePatient/lookahead_pytorch
        """
        if not 0.0 <= alpha <= 1.0:
            raise ValueError(f"Invalid slow update rate: {alpha}")
        if not 1 <= k:
            raise ValueError(f"Invalid lookahead steps: {k}")
        self.optimizer = base_optimizer
        self.param_groups = self.optimizer.param_groups
        self.alpha = alpha
        self.k = k
        for group in self.param_groups:
            group["step_counter"] = 0
        self.slow_weights = [
            [p.clone().detach() for p in group["params"]]
            for group in self.param_groups]

        for w in itertools.chain(*self.slow_weights):
            w.requires_grad = False

    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()
        loss = self.optimizer.step()
        for group, slow_weights in zip(self.param_groups, self.slow_weights):
            group["step_counter"] += 1
            if group["step_counter"] % self.k != 0:
                continue
            for p, q in zip(group["params"], slow_weights):
                if p.grad is None:
                    continue
                q.data.add_(self.alpha, p.data - q.data)
                p.data.copy_(q.data)
        return loss

    @classmethod
    def get_from_params(
        cls,
        params: Dict,
        base_optimizer_params: Dict = None,
        **kwargs,
    ) -> "Lookahead":
        from catalyst.dl.registry import OPTIMIZERS

        base_optimizer = OPTIMIZERS.get_from_params(
            params=params, **base_optimizer_params)
        optimizer = cls(base_optimizer=base_optimizer, **kwargs)
        return optimizer
