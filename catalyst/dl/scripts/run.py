#!/usr/bin/env python

import argparse
from argparse import ArgumentParser
from pathlib import Path

import safitty

from catalyst.dl import utils


def build_args(parser: ArgumentParser):
    """Constructs the command-line arguments for ``catalyst-dl run``"""
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
    utils.boolean_flag(parser, "verbose", default=None)
    utils.boolean_flag(parser, "check", default=None)
    utils.boolean_flag(
        parser, "deterministic",
        default=None,
        help="Deterministic mode if running in CuDNN backend"
    )
    utils.boolean_flag(
        parser, "benchmark",
        default=None,
        help="Use CuDNN benchmark"
    )

    return parser


def parse_args():
    """Parses the command line arguments and returns arguments and config"""
    parser = argparse.ArgumentParser()
    build_args(parser)
    args, unknown_args = parser.parse_known_args()
    return args, unknown_args


def main(args, unknown_args):
    """Run the ``catalyst-dl run`` script"""
    args, config = utils.parse_args_uargs(args, unknown_args)
    utils.set_global_seed(args.seed)
    utils.prepare_cudnn(args.deterministic, args.benchmark)

    Experiment, Runner = utils.import_experiment_and_runner(Path(args.expdir))

    runner_params = config.pop("runner_params", {}) or {}
    experiment = Experiment(config)
    runner = Runner(**runner_params)

    if experiment.logdir is not None:
        utils.dump_environment(config, experiment.logdir, args.configs)
        utils.dump_code(args.expdir, experiment.logdir)

    check_run = safitty.get(config, "args", "check", default=False)
    runner.run_experiment(experiment, check=check_run)


if __name__ == "__main__":
    args, unknown_args = parse_args()
    main(args, unknown_args)
