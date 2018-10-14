from .linknet import LinkNet
from .unet import UNet
from .unet_neptune import UNetDenseNet, UNetSeResNetXt, UNetSeResNet, UNetResNet

models = {
    'unet': UNet,
    'linknet': LinkNet,
    'UNetDenseNet': UNetDenseNet,
    'UNetSeResNetXt': UNetSeResNetXt,
    'UNetSeResNet': UNetSeResNet,
    'UNetResNet': UNetResNet,
}
