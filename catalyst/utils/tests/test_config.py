# flake8: noqa
import argparse
import pytest

from .. import config


def test_parse_config_args():
    configuration = {
        "stages": {
            "one": "uno",
            "two": "dos",
            "three": "tres"
        },
        "key": {
            "value": "key2"
        }
    }

    parser = argparse.ArgumentParser()
    parser.add_argument("--command")

    args, uargs = parser.parse_known_args(
        [
            "--command", "run", "--path=test.yml:str",
            "--stages/zero=cero:str", "-C=like:str"
        ]
    )

    configuration, args = config.parse_config_args(
        config=configuration, args=args, unknown_args=uargs
    )

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
