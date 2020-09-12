#!/usr/bin/env python
from typing import Dict, Tuple
import argparse
from argparse import ArgumentParser
import os
from pathlib import Path

import optuna

from catalyst.dl import utils


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

    parser.add_argument("--n-trials", type=int, default=None)
    parser.add_argument("--timeout", type=int, default=None)

    return parser


def parse_args():
    """Parses the command line arguments and returns arguments and config."""
    parser = argparse.ArgumentParser()
    build_args(parser)
    args, unknown_args = parser.parse_known_args()
    return args, unknown_args


def process_trial_config(trial, config: Dict) -> Tuple[optuna.Trial, Dict]:
    def _eval_trial_suggestions(x):
        nonlocal trial
        if isinstance(x, str) and "trial.suggest_" in x:
            x = eval(x)
        return x

    config = utils.maybe_recursive_call(config, _eval_trial_suggestions)
    return trial, config


def main_worker(args, unknown_args):
    """Runs main worker thread from model training."""
    args, config = utils.parse_args_uargs(args, unknown_args)
    utils.set_global_seed(args.seed)
    utils.prepare_cudnn(args.deterministic, args.benchmark)

    config.setdefault("distributed_params", {})["apex"] = args.apex
    config.setdefault("distributed_params", {})["amp"] = args.amp
    expdir = Path(args.expdir)

    def objective(trial: optuna.trial):
        trial, trial_config = process_trial_config(trial, config.copy())
        experiment, runner, trial_config = utils.prepare_config_api_components(
            expdir=expdir, config=trial_config
        )

        if experiment.logdir is not None and utils.get_rank() <= 0:
            utils.dump_environment(
                trial_config, experiment.logdir, args.configs
            )
            utils.dump_code(args.expdir, experiment.logdir)

        runner.run_experiment(experiment)

        return runner.best_valid_metrics[runner.main_metric]

    # @TODO: add more customization for optuna.Study and study.optimize
    direction = (
        "minimize"
        if config.get("stages", {})
        .get("stage_params", {})
        .get("minimize_metric", True)
        else "maximize"
    )
    study = optuna.create_study(
        direction=direction,
        # pruner=optuna.pruners.MedianPruner(
        #     n_startup_trials=1, n_warmup_steps=0, interval_steps=1
        # ),
    )
    study.optimize(objective, n_trials=args.n_trials, timeout=args.timeout)


def main(args, unknown_args):
    """Runs the ``catalyst-dl tune`` script."""
    main_worker(args, unknown_args)
    # utils.distributed_cmd_run(
    #     main_worker, args.distributed, args, unknown_args
    # )


if __name__ == "__main__":
    args, unknown_args = parse_args()
    main(args, unknown_args)
