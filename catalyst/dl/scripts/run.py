#!/usr/bin/env python

import argparse
from argparse import ArgumentParser
import os
from pathlib import Path

from catalyst.dl import utils
from catalyst.registry import EXPERIMENTS, RUNNERS
from catalyst.utils import distributed_cmd_run, get_rank


def build_args(parser: ArgumentParser):
    """Constructs the command-line arguments for ``catalyst-dl run``."""
    parser.add_argument(
        "--config",
        "--configs",
        "-C",
        nargs="+",
        help="path to config/configs",
        metavar="CONFIG_PATH",
        dest="configs",
        required=True,
    )
    parser.add_argument("--expdir", type=str, default=None)
    parser.add_argument("--logdir", type=str, default=None)
    parser.add_argument("--baselogdir", type=str, default=None)
    parser.add_argument(
        "-j",
        "--num-workers",
        default=None,
        type=int,
        help="number of data loading workers",
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
        help="path to latest checkpoint",
    )
    parser.add_argument(
        "--autoresume",
        type=str,
        help=(
            "try automatically resume from logdir//{best,last}_full.pth "
            "if --resume is empty"
        ),
        required=False,
        choices=["best", "last"],
        default=None,
    )
    parser.add_argument("--seed", type=int, default=42)
    utils.boolean_flag(
        parser,
        "apex",
        default=os.getenv("USE_APEX", "0") == "1",
        help="Enable/disable using of Apex extension",
    )
    utils.boolean_flag(
        parser,
        "amp",
        default=os.getenv("USE_AMP", "0") == "1",
        help="Enable/disable using of PyTorch AMP extension",
    )
    utils.boolean_flag(
        parser,
        "distributed",
        shorthand="ddp",
        default=os.getenv("USE_DDP", "0") == "1",
        help="Run in distributed mode",
    )
    utils.boolean_flag(parser, "verbose", default=None)
    utils.boolean_flag(parser, "timeit", default=None)
    utils.boolean_flag(parser, "check", default=None)
    utils.boolean_flag(parser, "overfit", default=None)
    utils.boolean_flag(
        parser,
        "deterministic",
        default=None,
        help="Deterministic mode if running in CuDNN backend",
    )
    utils.boolean_flag(
        parser, "benchmark", default=None, help="Use CuDNN benchmark"
    )

    return parser


def parse_args():
    """Parses the command line arguments and returns arguments and config."""
    parser = argparse.ArgumentParser()
    build_args(parser)
    args, unknown_args = parser.parse_known_args()
    return args, unknown_args


def main_worker(args, unknown_args):
    """Runs main worker thread from model training."""
    args, config = utils.parse_args_uargs(args, unknown_args)
    utils.set_global_seed(args.seed)
    utils.prepare_cudnn(args.deterministic, args.benchmark)

    config.setdefault("distributed_params", {})["apex"] = args.apex
    config.setdefault("distributed_params", {})["amp"] = args.amp

    experiment_fn, runner_fn = utils.import_experiment_and_runner(
        Path(args.expdir)
    )

    experiment_params = config.get("experiment_params", {})
    experiment_from_config = experiment_params.pop("experiment", None)
    assert any(
        x is None for x in (experiment_fn, experiment_from_config)
    ), "Experiment is set both in code and config."
    if experiment_fn is None and experiment_from_config is not None:
        experiment_fn = EXPERIMENTS.get(experiment_from_config)

    runner_params = config.get("runner_params", {})
    runner_from_config = runner_params.pop("runner", None)
    assert any(
        x is None for x in (runner_fn, runner_from_config)
    ), "Runner is set both in code and config."
    if runner_fn is None and runner_from_config is not None:
        runner_fn = RUNNERS.get(runner_from_config)

    experiment = experiment_fn(config)
    runner = runner_fn(**runner_params)

    if experiment.logdir is not None and get_rank() <= 0:
        utils.dump_environment(config, experiment.logdir, args.configs)
        utils.dump_code(args.expdir, experiment.logdir)

    runner.run_experiment(experiment)


def main(args, unknown_args):
    """Runs the ``catalyst-dl run`` script."""
    distributed_cmd_run(main_worker, args.distributed, args, unknown_args)


if __name__ == "__main__":
    args, unknown_args = parse_args()
    main(args, unknown_args)
