import numpy as np
import torch


def compute_mixup_lambda(bs, alpha, share_lambda=True):
    if share_lambda:
        lambda_ = np.random.beta(alpha, alpha)
    else:
        lambda_ = np.random.beta(alpha, alpha, (bs, 1)).astype(np.float32)
    return lambda_


def mixup(x, lambda_=None, alpha=None, share_lambda=True):
    assert any([x is not None for x in [lambda_, alpha]])

    perturb = x.copy()
    cats = [perturb[1:, ...], perturb[0:1, ...]]
    perturb = np.concatenate(cats, axis=0)

    if lambda_ is None:
        bs = x.shape[0]
        lambda_ = compute_mixup_lambda(bs, alpha, share_lambda)

    x = lambda_ * x + (1.0 - lambda_) * perturb
    return x


def mixup_torch(x, lambda_=None, alpha=None, share_lambda=True):
    assert any([x is not None for x in [lambda_, alpha]])

    perturb = x.clone()
    cats = [perturb[1:, ...], perturb[0:1, ...]]
    perturb = torch.cat(cats, dim=0).detach()

    if lambda_ is None:
        bs = x.shape[0]
        lambda_ = compute_mixup_lambda(bs, alpha, share_lambda)

    x = lambda_ * x + (1.0 - lambda_) * perturb
    return x
