import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    def __init__(self, gamma=2):
        super().__init__()
        self.gamma = gamma

    # TODO refactor
    def forward(self, outputs, target):
        if target.size() != outputs.size():
            raise ValueError(
                f"Targets and inputs must be same size. Got ({target.size()}) and ({outputs.size()})"
            )

        max_val = (-outputs).clamp(min=0)
        loss = outputs - outputs * target + max_val + \
               ((-max_val).exp() + (-outputs - max_val).exp()).log()

        invprobs = F.logsigmoid(-outputs * (target * 2.0 - 1.0))
        loss = (invprobs * self.gamma).exp() * loss

        return loss.sum(dim=1).mean()
