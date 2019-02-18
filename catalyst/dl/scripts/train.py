#!/usr/bin/env python

import os
import argparse
from importlib.util import spec_from_file_location, module_from_spec
from pathlib import Path
import sys

from catalyst.utils.config import prepare_config
from catalyst.utils.misc import set_global_seeds


def build_args(parser):
    parser.add_argument("--config", required=True)
    parser.add_argument("--expdir", type=Path, default=None)

    return parser


def import_experiment_and_runner(exp_dir: Path):
    s = spec_from_file_location(
        exp_dir.name, str(exp_dir / '__init__.py'),
        submodule_search_locations=[exp_dir.absolute()]
    )
    m = module_from_spec(s)
    sys.modules[exp_dir.name] = m
    s.loader.exec_module(m)
    return m.Experiment, m.Runner


def parse_args():
    parser = argparse.ArgumentParser()
    build_args(parser)
    args, unknown_args = parser.parse_known_args()
    return args, unknown_args


def main(args, unknown_args):
    config = prepare_config(args.config, unknown_args)

    set_global_seeds(config.get('seed', 42))

    Experiment, Runner = import_experiment_and_runner(args.expdir)

    e = Experiment(config)
    runner = Runner(e)

    runner.train()


if __name__ == "__main__":
    args, unknown_args = parse_args()
    main(args, unknown_args)
