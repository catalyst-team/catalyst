# flake8: noqa

from io import BytesIO
from pathlib import Path

import cv2
import numpy as np
import pytest
from unittest.mock import patch

from catalyst.utils.tensorboard import \
    EventsFileReader, EventReadingError, SummaryReader


def _get_test_data():
    """Test events file

    tag  value         step
    -----------------------
    x     1.0          1
    y    -1.0          1
    x     2.0          2
    z     zeros 2×2×3  1

    The first event is empty with wall_time = 1557489465

    log.add_scalar("x", 1.0, global_step=1)
    log.add_scalar("y", -1.0, global_step=1)
    log.add_scalar("x", 2.0, global_step=2)
    log.add_image("z", np.zeros((2, 2, 1)), global_step=1)
    """

    data_raw = [
        None,
        {
            "tag": "x",
            "value": 1.0,
            "step": 1,
            "type": "scalar"
        },
        {
            "tag": "y",
            "value": -1.0,
            "step": 1,
            "type": "scalar"
        },
        {
            "tag": "x",
            "value": 2.0,
            "step": 2,
            "type": "scalar"
        },
        {
            "tag": "z",
            "value": np.zeros((2, 2, 3)),
            "height": 2,
            "width": 2,
            "channels": 3,
            "step": 1,
            "type": "image"
        },
    ]
    # noqa: Q000
    data = b'\t\x00\x00\x00\x00\x00\x00\x007\xf9q9\t\xc9\xebE\x18`5' \
           b'\xd7A\x04A\xf4n\x17\x00\x00\x00\x00\x00\x00\x00\xe7\xce' \
           b'\xf8\x1e\t=\x82{\x19`5\xd7A\x10\x01*\n\n\x08\n\x01x\x15' \
           b'\x00\x00\x80?c\xf1\xd84\x17\x00\x00\x00\x00\x00\x00\x00' \
           b'\xe7\xce\xf8\x1e\tT\xe4g\x1a`5\xd7A\x10\x01*\n\n\x08\n' \
           b'\x01y\x15\x00\x00\x80\xbf{;wp\x17\x00\x00\x00\x00\x00\x00' \
           b'\x00\xe7\xce\xf8\x1e\t"S\xbc\x1b`5\xd7A\x10\x02*\n\n\x08' \
           b'\n\x01x\x15\x00\x00\x00@\x1d\xb9\xdc\x83`\x00\x00\x00\x00' \
           b'\x00\x00\x00(!\xc6\xda\t.\x03H"`5\xd7A\x10\x01*S\nQ\n\x01' \
           b'z"L\x08\x02\x10\x02\x18\x03"D\x89PNG\r\n\x1a\n\x00\x00\x00' \
           b'\rIHDR\x00\x00\x00\x02\x00\x00\x00\x02\x08\x02\x00\x00\x00' \
           b'\xfd\xd4\x9as\x00\x00\x00\x0bIDATx\x9cc`@\x06\x00\x00\x0e' \
           b'\x00\x01\xa9\x91s\xb1\x00\x00\x00\x00IEND\xaeB`\x82\x96x9j'
    return data, data_raw


def _compare_image_data(png, data):
    png_buf = np.frombuffer(png, np.uint8)
    png_decoded = cv2.imdecode(png_buf, cv2.IMREAD_COLOR)
    assert np.all(png_decoded == data), "Corrupted image data"


def test_events_reader_successful():
    data, data_raw = _get_test_data()
    reader = EventsFileReader(BytesIO(data))
    for event, event_raw in zip(reader, data_raw):
        if event_raw is not None:
            assert event.step == event_raw["step"]
            assert event.HasField("summary")
            assert len(event.summary.value) == 1
            if event_raw["type"] == "scalar":
                assert event.summary.value[0].HasField("simple_value")
                assert event.summary.value[0].tag == event_raw["tag"]
                assert event.summary.value[0].simple_value == event_raw["value"
                                                                        ]
            elif event_raw["type"] == "image":
                assert event.summary.value[0].HasField("image")
                assert event.summary.value[0].image.height == 2
                assert event.summary.value[0].image.width == 2
                assert event.summary.value[0].image.colorspace == 3
                _compare_image_data(
                    event.summary.value[0].image.encoded_image_string,
                    event_raw["value"]
                )


def test_events_reader_empty():
    data = BytesIO(b"")
    reader = EventsFileReader(data)
    assert len(list(reader)) == 0


def test_events_reader_invalid_data():
    data, _ = _get_test_data()
    data1 = bytearray(data)
    data1[0] = (data1[0] + 1) % 256
    reader = EventsFileReader(BytesIO(data1))
    with pytest.raises(EventReadingError):
        list(reader)

    data2 = bytearray(data)
    data2[123] = (data2[123] + 1) % 256
    reader = EventsFileReader(BytesIO(data2))
    with pytest.raises(EventReadingError):
        list(reader)


def test_events_reader_unexpected_end():
    data, _ = _get_test_data()
    data = data[:-5]
    reader = EventsFileReader(BytesIO(data))
    with pytest.raises(EventReadingError):
        list(reader)


def _open(path, mode):
    data, _ = _get_test_data()
    return BytesIO(data)


@patch("pathlib.Path.glob", lambda s, p: [Path("1"), Path("2")])
@patch("pathlib.Path.is_file", lambda s: True)
@patch("builtins.open", _open)
def test_summary_reader_iterate():
    reader = SummaryReader("logs", types=["scalar", "image"])
    _, data_raw = _get_test_data()
    data_raw2 = 2 * [d for d in data_raw if d is not None]
    items = list(reader)

    assert len(items) == len(data_raw2)

    for item, event_raw in zip(items, data_raw2):
        assert item.step == event_raw["step"]
        assert item.tag == event_raw["tag"]
        assert item.type == event_raw["type"]
        assert np.all(item.value == event_raw["value"])


@patch("pathlib.Path.glob", lambda s, p: [Path("1"), Path("2")])
@patch("pathlib.Path.is_file", lambda s: True)
@patch("builtins.open", _open)
def test_summary_reader_filter():
    tags = ["x", "z"]
    reader = SummaryReader("logs", tag_filter=tags, types=["scalar", "image"])
    _, data_raw = _get_test_data()
    data_raw2 = 2 * [d for d in data_raw if d is not None and d["tag"] in tags]
    items = list(reader)

    assert len(items) == len(data_raw2)

    for item, event_raw in zip(items, data_raw2):
        assert item.step == event_raw["step"]
        assert item.tag == event_raw["tag"]
        assert item.type == event_raw["type"]
        assert item.tag in tags
        assert np.all(item.value == event_raw["value"])


@patch("pathlib.Path.glob", lambda s, p: [Path("1"), Path("2")])
@patch("pathlib.Path.is_file", lambda s: True)
@patch("builtins.open", _open)
def test_summary_reader_filter_scalars():
    types = ["scalar"]
    reader = SummaryReader("logs", types=types)
    _, data_raw = _get_test_data()
    data_raw2 = 2 * [
        d for d in data_raw if d is not None and d["type"] in types
    ]
    items = list(reader)

    assert len(items) == len(data_raw2)

    for item, event_raw in zip(items, data_raw2):
        assert item.step == event_raw["step"]
        assert item.tag == event_raw["tag"]
        assert item.type == "scalar"
        assert np.all(item.value == event_raw["value"])


def test_summary_reader_invalid_type():
    with pytest.raises(ValueError):
        SummaryReader(".", types=["unknown-type"])
