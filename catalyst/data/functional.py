import os
import numpy as np
import cv2
import jpeg4py as jpeg
import torch


def read_image(image_name, datapath=None, grayscale=False):
    if datapath is not None:
        image_name = (
            image_name if image_name.startswith(datapath) else
            os.path.join(datapath, image_name)
        )

    img = None
    try:
        if image_name.endswith(("jpg", "JPG", "jpeg", "JPEG")):
            img = jpeg.JPEG(image_name).decode()
    except Exception:
        pass

    if img is None:
        img = cv2.imread(image_name)

        if len(img.shape) == 3:  # BGR -> RGB
            img = img[:, :, ::-1]

    if len(img.shape) < 3:  # grayscale
        img = np.expand_dims(img, -1)

    if img.shape[-1] != 3 and not grayscale:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)

    return img


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
