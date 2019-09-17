import os
import logging
import base64
import numpy as np
from six import string_types
from catalyst.utils.serialization import serialize, deserialize

logger = logging.getLogger(__name__)

if os.environ.get("USE_LZ4", "1") == "1":
    try:
        import lz4.frame
        LZ4_ENABLED = True
    except ImportError:
        logger.warning(
            "lz4 not available, disabling compression. "
            "To install lz4, run `pip install lz4`."
        )
        LZ4_ENABLED = False
else:
    LZ4_ENABLED = False


def is_compressed(data):
    return isinstance(data, bytes) or isinstance(data, string_types)


def compress(data):
    if LZ4_ENABLED:
        data = serialize(data)
        data = lz4.frame.compress(data)
        data = base64.b64encode(data).decode("ascii")
    return data


def compress_if_needed(data):
    if isinstance(data, np.ndarray):
        data = compress(data)
    return data


def decompress(data):
    if LZ4_ENABLED:
        data = base64.b64decode(data)
        data = lz4.frame.decompress(data)
        data = deserialize(data)
    return data


def decompress_if_needed(data):
    if is_compressed(data):
        data = decompress(data)
    return data


if LZ4_ENABLED:
    pack = compress
    pack_if_needed = compress_if_needed
    unpack = decompress
    unpack_if_needed = decompress_if_needed
else:
    pack = serialize
    pack_if_needed = serialize
    unpack = deserialize
    unpack_if_needed = deserialize
