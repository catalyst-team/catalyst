# flake8: noqa
import logging

from torch.jit.frontend import UnsupportedNodeError

from catalyst.settings import SETTINGS

from catalyst.contrib.data.cv.transforms.torch import (
    Compose,
    Normalize,
    ToTensor,
    normalize,
    to_tensor,
)

logger = logging.getLogger(__name__)


try:
    from catalyst.contrib.data.cv.reader import ImageReader, MaskReader
    from catalyst.contrib.data.cv.dataset import ImageFolderDataset

    from catalyst.contrib.data.cv.mixins import (
        BlurMixin,
        FlareMixin,
        RotateMixin,
    )

    from catalyst.contrib.data.cv.transforms.albumentations import (
        TensorToImage,
        ImageToTensor,
    )
    from catalyst.contrib.data.cv.transforms.kornia import (
        OneOfPerBatch,
        OneOfPerSample,
    )
except ImportError as ex:
    if SETTINGS.cv_required:
        logger.warning(
            "some of catalyst-cv dependencies are not available,"
            " to install dependencies, run `pip install catalyst[cv]`."
        )
        raise ex
except UnsupportedNodeError as ex:
    logger.warning(
        "kornia has requirement torch>=1.6.0,"
        " probably you have an old version of torch which is incompatible.\n"
        "To update pytorch, run `pip install -U 'torch>=1.6.0'`."
    )
    if SETTINGS.kornia_required:
        raise ex
