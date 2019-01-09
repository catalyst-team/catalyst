import torch

from . import unet as unet_loss
from . import center_loss
from . import contrastive as contrastive_loss
from . import huber as huber_loss
from . import ce
from . import bcece
from . import focal_loss
from . import dice

CRITERION = {
    **torch.nn.__dict__,
    **unet_loss.__dict__,
    **center_loss.__dict__,
    **contrastive_loss.__dict__,
    **huber_loss.__dict__,
    **ce.__dict__,
    **bcece.__dict__,
    **focal_loss.__dict__,
    **dice.__dict__,
}
