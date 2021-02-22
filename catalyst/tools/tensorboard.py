from typing import BinaryIO, Optional, Union
from collections import namedtuple
from collections.abc import Iterable
from pathlib import Path
import struct

import numpy as np
from tensorboardX import SummaryWriter as tensorboardX_SummaryWriter
from tensorboardX.crc32c import crc32c
from tensorboardX.proto.event_pb2 import Event

SummaryWriter = tensorboardX_SummaryWriter


def _u32(x):
    return x & 0xFFFFFFFF


def _masked_crc32c(data):
    x = _u32(crc32c(data))
    return _u32(((x >> 15) | _u32(x << 17)) + 0xA282EAD8)


class EventReadingException(Exception):
    """An exception that correspond to an event file reading error."""


class EventsFileReader(Iterable):
    """An iterator over a Tensorboard events file."""

    def __init__(self, events_file: BinaryIO):
        """Initialize an iterator over an events file.

        Args:
            events_file: An opened file-like object.
        """
        self._events_file = events_file

    def _read(self, size: int) -> Optional[bytes]:
        """Read exactly next `size` bytes from the current stream.

        Args:
            size: A size in bytes to be read.

        Returns:
            A `bytes` object with read data or `None` on EOF.

        """
        data = self._events_file.read(size)
        if data is None:
            raise NotImplementedError("Reading of a stream in non-blocking mode")
        if 0 < len(data) < size:
            raise EventReadingException("The size of read data is less than requested size")
        if len(data) == 0:
            return None
        return data

    def _read_and_check(self, size: int) -> Optional[bytes]:
        """Read and check data described by a format string.

        Args:
            size:  A size in bytes to be read.

        Returns:
             A decoded number.
        """
        data = self._read(size)
        if data is None:
            return None
        checksum_size = struct.calcsize("I")
        checksum = struct.unpack("I", self._read(checksum_size))[0]
        checksum_computed = _masked_crc32c(data)
        if checksum != checksum_computed:
            raise EventReadingException(
                "Invalid checksum. {checksum} != {crc32}".format(
                    checksum=checksum, crc32=checksum_computed
                )
            )
        return data

    def __iter__(self) -> Event:
        """Iterates over events in the current events file.

        Returns:
            An Event object
        """
        while True:
            header_size = struct.calcsize("Q")
            header = self._read_and_check(header_size)
            if header is None:
                break
            event_size = struct.unpack("Q", header)[0]
            event_raw = self._read_and_check(event_size)
            if event_raw is None:
                raise EventReadingException("Unexpected end of events file")
            event = Event()
            event.ParseFromString(event_raw)
            yield event


SummaryItem = namedtuple("SummaryItem", ["tag", "step", "wall_time", "value", "type"])


def _get_scalar(value) -> Optional[np.ndarray]:
    """Decode an scalar event.

    Args:
        value: A value field of an event

    Returns:
        Decoded scalar
    """
    if value.HasField("simple_value"):
        return value.simple_value
    return None


class SummaryReader(Iterable):
    """Iterates over events in all the files in the current logdir.

    .. note::
        Only scalars are supported at the moment.
    """

    _DECODERS = {  # noqa: WPS115
        "scalar": _get_scalar,
    }

    def __init__(
        self,
        logdir: Union[str, Path],
        tag_filter: Optional[Iterable] = None,
        types: Iterable = ("scalar",),
    ):
        """Initalize new summary reader.

        Args:
            logdir: A directory with Tensorboard summary data
            tag_filter: A list of tags to leave (`None` for all)
            types: A list of types to get.
            Only "scalar" and "image" types are allowed at the moment.
        """
        self._logdir = Path(logdir)

        self._tag_filter = set(tag_filter) if tag_filter is not None else None
        self._types = set(types)
        self._check_type_names()

    def _check_type_names(self):
        if self._types is None:
            return
        if not all(type_name in self._DECODERS.keys() for type_name in self._types):
            raise ValueError("Invalid type name")

    def _decode_events(self, events: Iterable) -> Optional[SummaryItem]:
        """
        Convert events to `SummaryItem` instances.
        Returns a generator with decoded events
        or `None` if an event can't be decoded.

        Args:
            events: An iterable with events objects.

        Returns:
            Optional[SummaryItem]: decoded event
        """
        for event in events:
            if not event.HasField("summary"):
                yield None
            step = event.step
            wall_time = event.wall_time
            for value in event.summary.value:
                tag = value.tag
                for value_type in self._types:
                    decoder = self._DECODERS[value_type]
                    data = decoder(value)
                    if data is not None:
                        yield SummaryItem(
                            tag=tag, step=step, wall_time=wall_time, value=data, type=value_type,
                        )
                else:
                    yield None

    def _check_tag(self, tag: str) -> bool:
        """Check if a tag matches the current tag filter.

        Args:
            tag: A string with tag

        Returns:
            A boolean value.
        """
        return self._tag_filter is None or tag in self._tag_filter

    def __iter__(self) -> SummaryItem:
        """Iterate over events in all the files in the current logdir.

        Returns:
            A generator with `SummaryItem` objects
        """
        log_files = sorted(f for f in self._logdir.glob("*") if f.is_file())
        for file_path in log_files:
            with open(file_path, "rb") as f:
                reader = EventsFileReader(f)
                yield from (
                    item
                    for item in self._decode_events(reader)
                    if item is not None and self._check_tag(item.tag) and item.type in self._types
                )


# __all__ = [
#     "EventReadingException",
#     "EventsFileReader",
#     "SummaryItem",
#     "SummaryReader",
#     "SummaryWriter",
# ]
