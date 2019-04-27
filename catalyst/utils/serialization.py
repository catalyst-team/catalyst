import pyarrow


def serialize(x):
    """
    Serialize the data into bytes using marshal and zlib

    Args:
        x: a value

    Returns:
        Returns a bytes object containing compressed with zlib data.
    """
    return pyarrow.serialize(x).to_buffer().to_pybytes()


def deserialize(serialized_x):
    """
    Deserialize bytes into an object using zlib and marshal

    Args:
        bytes: a bytes object containing compressed with zlib data

    Returns:
        Returns a value decompressed from the bytes-like object.
    """
    return pyarrow.deserialize(serialized_x)
