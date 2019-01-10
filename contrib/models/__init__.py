from .resnet_encoder import ResnetEncoder
from .sequential import SequentialNet
from . import segmentation

__all__ = ["ResnetEncoder", "SequentialNet"]

MODELS = {
    **{
        "ResnetEncoder": ResnetEncoder,
        "SequentialNet": SequentialNet
    },
    **segmentation.__dict__
}
