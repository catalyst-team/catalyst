import zlib
import marshal


def serialize(object):
    return zlib.compress(marshal.dumps(object, 2))


def deserialize(bytes):
    return marshal.loads(zlib.decompress(bytes))
