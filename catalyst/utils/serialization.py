import pyarrow


def serialize(data):
    """
    Serialize the data into bytes using pyarrow

    Args:
        data: a value

    Returns:
        Returns a bytes object serialized with pyarrow data.
    """
    return pyarrow.serialize(data).to_buffer().to_pybytes()


def deserialize(data):
    """
    Deserialize bytes into an object using pyarrow

    Args:
        bytes: a bytes object containing serialized with pyarrow data.

    Returns:
        Returns a value deserialized from the bytes-like object.
    """
    return pyarrow.deserialize(data)
