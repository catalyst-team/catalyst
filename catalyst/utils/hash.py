from typing import Any
from base64 import urlsafe_b64encode
from hashlib import sha256


def _make_hashable(o):
    if isinstance(o, (tuple, list)):
        return tuple(((type(o).__name__, _make_hashable(e)) for e in o))
    if isinstance(o, dict):
        return tuple(
            sorted(
                (type(o).__name__, k, _make_hashable(v)) for k, v in o.items()
            )
        )
    if isinstance(o, (set, frozenset)):
        return tuple(sorted((type(o).__name__, _make_hashable(e)) for e in o))
    return o


def get_hash(obj: Any) -> str:
    """
    Creates unique hash from object following way:
    - Represent obj as sting recursively
    - Hash this string with sha256 hash function
    - encode hash with url-safe base64 encoding

    Args:
        obj: object to hash

    Returns:
        base64-encoded string
    """
    bytes_to_hash = repr(_make_hashable(obj)).encode()
    hash_bytes = sha256(bytes_to_hash).digest()
    return urlsafe_b64encode(hash_bytes).decode()


def get_short_hash(o) -> str:
    """
    @TODO: Docs. Contribution is welcome.
    """
    hash = get_hash(o)[:6]
    return hash


__all__ = ["get_hash", "get_short_hash"]
