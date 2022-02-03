# flake8: noqa

from catalyst.settings import SETTINGS

from catalyst.contrib.models.mnist import MnistBatchNormNet, MnistSimpleNet

if SETTINGS.cv_required:
    from catalyst.contrib.models.resnet_encoder import ResnetEncoder
