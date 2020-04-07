import pytest

from .. import __main__ as main


def test_arg_parser_run():
    """@TODO: Docs. Contribution is welcome."""
    parser = main.build_parser()

    args, uargs = parser.parse_known_args(
        ["run", "--config", "test.yml", "--unknown"]
    )

    assert args.command == "run"
    assert args.configs == ["test.yml"]
    assert "--unknown" in uargs


def test_run_multiple_configs():
    """@TODO: Docs. Contribution is welcome."""
    parser = main.build_parser()

    args, uargs = parser.parse_known_args(
        ["run", "--config", "test.yml", "test1.yml"]
    )

    assert args.configs == ["test.yml", "test1.yml"]


def test_arg_parser_fail_on_none():
    """@TODO: Docs. Contribution is welcome."""
    parser = main.build_parser()

    with pytest.raises(SystemExit):
        # Raises SystemExit when args are not ok
        parser.parse_known_args(["--config", "test.yml", "--unknown"])
