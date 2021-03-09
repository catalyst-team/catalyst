# flake8: noqa
import argparse
import io
import json

import numpy as np

from catalyst.dl.scripts.misc import parse_config_args
from catalyst.utils.config import _load_ordered_yaml


def test_parse_config_args():
    configuration = {
        "stages": {"one": "uno", "two": "dos", "three": "tres"},
        "key": {"value": "key2"},
    }

    parser = argparse.ArgumentParser()
    parser.add_argument("--command")

    args, uargs = parser.parse_known_args(
        ["--command", "run", "--path=test.yml:str", "--stages/zero=cero:str", "-C=like:str"]
    )

    configuration, args = parse_config_args(config=configuration, args=args, unknown_args=uargs)

    assert args.command == "run"
    assert args.path == "test.yml"
    assert configuration.get("stages") is not None
    assert "zero" in configuration["stages"]
    assert configuration["stages"]["zero"] == "cero"
    assert configuration.get("args") is not None
    assert configuration["args"]["path"] == "test.yml"
    assert configuration["args"]["C"] == "like"
    assert configuration["args"]["command"] == "run"

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
    yaml_config = _load_ordered_yaml(buffer)

    for key, item in configuration.items():
        assert np.isclose(yaml_config[key], item)
