from .linknet import LinkNet
from .unet import UNet
from .resnetunet import ResNetUnet

models = {"unet": UNet, "resnetunet": ResNetUnet, "linknet": LinkNet}
