from .linknet import LinkNet
from .unet import UNet
from .unet_ternaus import AlbuNet, UNet11, UNet16

models = {
    'unet': UNet,
    'linknet': LinkNet,
    'unet11': UNet11,
    'unet16': UNet16,
    'albunet': AlbuNet
}
