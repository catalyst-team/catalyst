#!/usr/bin/env python

import argparse
from argparse import ArgumentParser
from pathlib import Path

from catalyst.utils import set_global_seed, boolean_flag, prepare_cudnn
from catalyst.utils.config import parse_args_uargs, dump_environment
from catalyst.utils.scripts import dump_code
from catalyst.dl.utils.scripts import import_experiment_and_runner


def build_args(parser: ArgumentParser):
    parser.add_argument(
        "--config",
        "--configs",
        "-C",
        nargs="+",
        help="path to config/configs",
        metavar="CONFIG_PATH",
        dest="configs",
        required=True
    )
    parser.add_argument("--expdir", type=str, default=None)
    parser.add_argument("--logdir", type=str, default=None)
    parser.add_argument("--baselogdir", type=str, default=None)
    parser.add_argument(
        "-j",
        "--num-workers",
        default=None,
        type=int,
        help="number of data loading workers"
    )
    parser.add_argument(
        "-b", "--batch-size", default=None, type=int, help="mini-batch size"
    )
    parser.add_argument(
        "-e", "--num-epochs", default=None, type=int, help="number of epochs"
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
    boolean_flag(
        parser, "deterministic",
        default=None,
        help="Deterministic mode if running in CuDNN backend"
    )
    boolean_flag(
        parser, "benchmark",
        default=None,
        help="Use CuDNN benchmark"
    )

    return parser


def parse_args():
    parser = argparse.ArgumentParser()
    build_args(parser)
    args, unknown_args = parser.parse_known_args()
    return args, unknown_args


def main(args, unknown_args):
    args, config = parse_args_uargs(args, unknown_args)
    set_global_seed(args.seed)
    prepare_cudnn(args.deterministic, args.benchmark)

    Experiment, Runner = import_experiment_and_runner(Path(args.expdir))

    experiment = Experiment(config)
    runner = Runner()

    if experiment.logdir is not None:
        dump_environment(config, experiment.logdir, args.configs)
        dump_code(args.expdir, experiment.logdir)

    runner.run_experiment(experiment, check=args.check)


if __name__ == "__main__":
    args, unknown_args = parse_args()
    main(args, unknown_args)
