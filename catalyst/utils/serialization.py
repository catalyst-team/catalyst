import os
import logging
import pickle

logger = logging.getLogger(__name__)

if os.environ.get("USE_PYARROW", "1") == "1":
    try:
        import pyarrow
        PYARROW_ENABLED = True
    except ImportError:
        logger.warning(
            "pyarrow not available, switching to pickle. "
            "To install pyarrow, run `pip install pyarrow`."
        )
        PYARROW_ENABLED = False
else:
    PYARROW_ENABLED = False


def pyarrow_serialize(data):
    """
    Serialize the data into bytes using pyarrow

    Args:
        data: a value

    Returns:
        Returns a bytes object serialized with pyarrow data.
    """
    return pyarrow.serialize(data).to_buffer().to_pybytes()


def pyarrow_deserialize(data):
    """
    Deserialize bytes into an object using pyarrow

    Args:
        bytes: a bytes object containing serialized with pyarrow data.

    Returns:
        Returns a value deserialized from the bytes-like object.
    """
    return pyarrow.deserialize(data)


def pickle_serialize(data):
    """
    Serialize the data into bytes using pickle

    Args:
        data: a value

    Returns:
        Returns a bytes object serialized with pickle data.
    """
    return pickle.dumps(data)


def pickle_deserialize(data):
    """
    Deserialize bytes into an object using pickle

    Args:
        bytes: a bytes object containing serialized with pickle data.

    Returns:
        Returns a value deserialized from the bytes-like object.
    """
    return pickle.loads(data)


if PYARROW_ENABLED:
    serialize = pyarrow_serialize
    deserialize = pyarrow_deserialize
else:
    serialize = pickle_serialize
    deserialize = pickle_deserialize
