# flake8: noqa
import argparse
import io
import json

import numpy as np

from catalyst import utils
from catalyst.utils import config


def test_parse_config_args():
    configuration = {
        "stages": {"one": "uno", "two": "dos", "three": "tres"},
        "key": {"value": "key2"},
    }

    parser = argparse.ArgumentParser()
    parser.add_argument("--command")
    parser.add_argument("--logdir", type=str, default=None)
    parser.add_argument(
        "--autoresume",
        type=str,
        required=False,
        choices=["best", "last"],
        default=None,
    )

    args, uargs = parser.parse_known_args(
        ["--command", "run", "--logdir", "logdir", "--autoresume", "last"]
    )

    configuration, args = utils.parse_config_args(
        config=configuration, args=args, unknown_args=uargs
    )

    assert args.command == "run"
    assert configuration.get("stages") is not None
    assert "logdir" in configuration["args"]
    assert configuration["args"]["logdir"] == "logdir"
    assert "autoresume" in configuration["args"]
    assert configuration["args"]["autoresume"] == "last"

    for key, value in args._get_kwargs():
        v = configuration["args"].get(key)
        assert v is not None
        assert v == value


def test_parse_numbers():
    configuration = {
        "a": 1,
        "b": 20,
        "c": 303e5,
        "d": -4,
        "e": -50,
        "f": -666e7,
        "g": 0.35,
        "h": 7.35e-5,
        "k": 8e-10,
    }

    buffer = io.StringIO()
    json.dump(configuration, buffer)
    buffer.seek(0)
    yaml_config = config._load_ordered_yaml(buffer)

    for key, item in configuration.items():
        assert np.isclose(yaml_config[key], item)
