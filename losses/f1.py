import torch
import torch.nn as nn
import torch.nn.functional as F


class F1Loss(nn.Module):
    def __init__(self):
        super(F1Loss, self).__init__()
        self.small_value = 1e-6

    def forward(self, logits, labels):
        beta = 1
        batch_size = logits.size()[0]
        p = F.sigmoid(logits)
        l = labels
        num_pos = torch.sum(p, 1) + self.small_value
        num_pos_hat = torch.sum(l, 1) + self.small_value
        tp = torch.sum(l * p, 1)
        precise = tp / num_pos
        recall = tp / num_pos_hat
        fs = (1 + beta * beta) * precise * recall / (beta * beta * precise + recall + self.small_value)
        loss = fs.sum() / batch_size
        return 1 - loss
