#!/usr/bin/env python

import os
import sys
import argparse
from importlib.util import spec_from_file_location, module_from_spec
from pathlib import Path

from catalyst.utils.config import parse_args_uargs
from catalyst.utils.misc import set_global_seeds, boolean_flag


def build_args(parser):
    parser.add_argument("--config", required=True)
    parser.add_argument("--expdir", type=Path, default=None)
    parser.add_argument(
        "-j",
        "--n-workers",
        default=None,
        type=int,
        help="number of data loading workers"
    )
    parser.add_argument(
        "-b", "--batch-size", default=None, type=int, help="mini-batch size"
    )
    parser.add_argument(
        "-e",
        "--n-epochs",
        default=None,
        type=int,
        help="number of epochs"
    )
    parser.add_argument(
        "--resume",
        default=None,
        type=str,
        metavar="PATH",
        help="path to latest checkpoint"
    )
    parser.add_argument("--seed", type=int, default=42)
    boolean_flag(parser, "verbose", default=False)
    boolean_flag(parser, "check", default=False)

    return parser


def parse_args():
    parser = argparse.ArgumentParser()
    build_args(parser)
    args, unknown_args = parser.parse_known_args()
    return args, unknown_args


def import_experiment_and_runner(exp_dir: Path):
    # @TODO: better PYTHONPATH handling
    sys.path.insert(0, str(exp_dir.absolute()))
    sys.path.insert(0, os.path.dirname(str(exp_dir.absolute())))
    s = spec_from_file_location(
        exp_dir.name,
        str(exp_dir.absolute() / "__init__.py"),
        submodule_search_locations=[exp_dir.absolute()]
    )
    m = module_from_spec(s)
    s.loader.exec_module(m)
    sys.modules[exp_dir.name] = m
    Experiment, Runner = m.Experiment, m.Runner
    return Experiment, Runner


def main(args, unknown_args):
    args, config = parse_args_uargs(args, unknown_args, dump_config=True)
    set_global_seeds(config.get("seed", 42))

    Experiment, Runner = import_experiment_and_runner(Path(args.expdir))

    experiment = Experiment(config)
    runner = Runner()

    runner.run(
        experiment,
        check=args.check
    )


if __name__ == "__main__":
    args, unknown_args = parse_args()
    main(args, unknown_args)
