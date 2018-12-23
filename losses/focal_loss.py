import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    def __init__(self, gamma=2):
        super().__init__()
        self.gamma = gamma

    # TODO refactor
    def forward(self, outputs, targets):
        if targets.size() != outputs.size():
            raise ValueError(
                f"Targets and inputs must be same size. "
                f"Got ({targets.size()}) and ({outputs.size()})"
            )

        max_val = (-outputs).clamp(min=0)
        log_ = ((-max_val).exp() + (-outputs - max_val).exp()).log()
        loss = outputs - outputs * targets + max_val + log_

        invprobs = F.logsigmoid(-outputs * (targets * 2.0 - 1.0))
        loss = (invprobs * self.gamma).exp() * loss

        return loss.sum(dim=1).mean()
