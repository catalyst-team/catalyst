# flake8: noqa
# @TODO: code formatting issue for 20.07 release
import base64
import logging

import numpy as np
from six import string_types

from catalyst.contrib.utils.serialization import deserialize, serialize
from catalyst.settings import SETTINGS

logger = logging.getLogger(__name__)

if SETTINGS.use_lz4:
    try:
        import lz4.frame
    except ImportError as ex:
        logger.warning("lz4 not available, to install lz4, run `pip install lz4`.")
        raise ex


def is_compressed(data):
    """@TODO: Docs. Contribution is welcome."""
    return isinstance(data, (bytes, string_types))


def compress(data):
    """@TODO: Docs. Contribution is welcome."""
    if SETTINGS.use_lz4:
        data = serialize(data)
        data = lz4.frame.compress(data)
        data = base64.b64encode(data).decode("ascii")
    return data


def compress_if_needed(data):
    """@TODO: Docs. Contribution is welcome."""
    if isinstance(data, np.ndarray):
        data = compress(data)
    return data


def decompress(data):
    """@TODO: Docs. Contribution is welcome."""
    if SETTINGS.use_lz4:
        data = base64.b64decode(data)
        data = lz4.frame.decompress(data)
        data = deserialize(data)
    return data


def decompress_if_needed(data):
    """@TODO: Docs. Contribution is welcome."""
    if is_compressed(data):
        data = decompress(data)
    return data


if SETTINGS.use_lz4:
    pack = compress
    pack_if_needed = compress_if_needed
    unpack = decompress
    unpack_if_needed = decompress_if_needed
else:
    pack = serialize
    pack_if_needed = serialize
    unpack = deserialize
    unpack_if_needed = deserialize

__all__ = ["pack", "pack_if_needed", "unpack", "unpack_if_needed"]
