from .linknet import LinkNet
from .unet import UNet
from .unet_ternaus import AlbuNet, UNet11, UNet16
from .unet_resnet import UNetResNet
from .unet_resnext import UNetSeResnext50, UNetSenet154

models = {
    'unet': UNet,
    'linknet': LinkNet,
    'unet11': UNet11,
    'unet16': UNet16,
    'albunet': AlbuNet,
    'unetresnet': UNetResNet,
    'unetresnext': UNetSeResnext50,
    'unetsenet154': UNetSenet154
}
