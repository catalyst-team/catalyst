import hashlib
import base64


def make_hash_sha256(o):
    hasher = hashlib.sha256()
    hasher.update(repr(make_hashable(o)).encode())
    return base64.b64encode(hasher.digest()).decode()


def make_hashable(o):
    if isinstance(o, (tuple, list)):
        return tuple(((type(o).__name__, make_hashable(e)) for e in o))
    if isinstance(o, dict):
        return tuple(
            sorted(
                (type(o).__name__, k, make_hashable(v)) for k, v in o.items()
            )
        )
    if isinstance(o, (set, frozenset)):
        return tuple(sorted((type(o).__name__, make_hashable(e)) for e in o))
    return o


def get_hash(o) -> str:
    hashable_o = make_hashable(o)
    hash = make_hash_sha256(hashable_o)
    return hash


def get_short_hash(o) -> str:
    hash = get_hash(o)[:6]
    return hash
