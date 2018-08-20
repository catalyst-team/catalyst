import torch
import common.losses.unet as unet_loss
import common.losses.center_loss as center_loss
import common.losses.contrastive as contrastive_loss
import common.losses.huber as huber_loss


LOSSES = {
    **torch.nn.__dict__,
    **unet_loss.__dict__,
    **center_loss.__dict__,
    **contrastive_loss.__dict__,
    **huber_loss.__dict__
}
