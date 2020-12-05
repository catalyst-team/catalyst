# flake8: noqa

import logging

logger = logging.getLogger(__name__)

from catalyst.settings import SETTINGS

try:
    from catalyst.contrib.utils.cv.image import (
        has_image_extension,
        imread,
        imwrite,
        imsave,
        mask_to_overlay_image,
        mimread,
        mimwrite_with_meta,
    )
except ImportError as ex:
    if SETTINGS.cv_required:
        logger.warning(
            "catalyst[cv] requirements are not available, to install them,"
            " run `pip install catalyst[cv]`."
        )
        raise ex

from catalyst.contrib.utils.cv.tensor import (
    tensor_from_rgb_image,
    tensor_to_ndimage,
)
