from .. import __main__ as main

import pytest


def test_arg_parser_train():
    parser = main.build_parser()

    args, uargs = parser.parse_known_args([
        'run',
        '--config', 'test.yml',
        '--unknown'
    ])

    assert args.command == 'run'
    assert args.config == 'test.yml'
    assert '--unknown' in uargs


def test_arg_parser_infer():
    parser = main.build_parser()

    args, uargs = parser.parse_known_args([
        'run',
        '--config', 'test.yml',
        '--unknown'
    ])

    assert args.command == 'run'
    assert args.config == 'test.yml'
    assert '--unknown' in uargs


def test_arg_parser_fail_on_none():
    parser = main.build_parser()

    with pytest.raises(SystemExit):
        # Raises SystemExit when args are not ok
        parser.parse_known_args([
            '--config', 'test.yml',
            '--unknown'
        ])
