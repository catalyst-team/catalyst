import zlib
import marshal


def serialize(object):
    """
    Serialize the data into bytes using marshal and zlib

    Args:
        object: a value

    Returns:
        Returns a bytes object containing compressed with zlib data.
    """
    return zlib.compress(marshal.dumps(object, 2))


def deserialize(bytes):
    """
    Deserialize bytes into an object using zlib and marshal

    Args:
        bytes: a bytes object containing compressed with zlib data

    Returns:
        Returns a value decompressed from the bytes-like object.
    """
    return marshal.loads(zlib.decompress(bytes))
