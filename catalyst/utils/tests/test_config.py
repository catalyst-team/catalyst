import argparse
import pytest

from .. import config


def test_parse_config_args():
    configuration = {
        "stages": {"one": "uno", "two": "dos", "three": "tres"},
        "key": {"value": "key2"}
    }

    parser = argparse.ArgumentParser()
    parser.add_argument("--command")

    args, uargs = parser.parse_known_args([
        "--command", "train",
        "--path=test.yml:str",
        "--stages/zero=cero:str"
    ])

    configuration, args = config.parse_config_args(config=configuration, args=args, unknown_args=uargs)
    assert args.command == "train"
    assert args.path == "test.yml"
    assert configuration.get("stages") is not None
    assert "zero" in configuration["stages"]
    assert configuration["stages"]["zero"] == "cero"
