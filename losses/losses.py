import torch
import catalyst.losses.unet as unet_loss
import catalyst.losses.center_loss as center_loss
import catalyst.losses.contrastive as contrastive_loss
import catalyst.losses.huber as huber_loss
import catalyst.losses.lovasz_losses as lovasz_losses
import catalyst.losses.focal_loss as focal_loss
import catalyst.losses.ce as ce

LOSSES = {
    **torch.nn.__dict__,
    **unet_loss.__dict__,
    **center_loss.__dict__,
    **contrastive_loss.__dict__,
    **huber_loss.__dict__,
    **lovasz_losses.__dict__,
    **focal_loss.__dict__,
	**ce.__dict__,
}
