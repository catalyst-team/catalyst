#!/usr/bin/env python
# Config API and Optuna integration for AutoML hyperparameters tuning.
from typing import Dict, Tuple
import argparse
from argparse import ArgumentParser
from pathlib import Path

import optuna

from catalyst.dl.scripts.misc import parse_args_uargs
from catalyst.runners.config import ConfigRunner
from catalyst.utils.distributed import get_rank
from catalyst.utils.misc import boolean_flag, maybe_recursive_call, set_global_seed
from catalyst.utils.sys import dump_code, dump_environment, get_config_runner
from catalyst.utils.torch import prepare_cudnn


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
    # parser.add_argument("--logdir", type=str, default=None)
    parser.add_argument("--baselogdir", type=str, default=None)
    # parser.add_argument(
    #     "-j", "--num-workers", default=None, type=int, help="number of data loading workers",
    # )
    # parser.add_argument("-b", "--batch-size", default=None, type=int, help="mini-batch size")
    # parser.add_argument("-e", "--num-epochs", default=None, type=int, help="number of epochs")
    # parser.add_argument(
    #     "--resume", default=None, type=str, metavar="PATH", help="path to latest checkpoint",
    # )
    # parser.add_argument(
    #     "--autoresume",
    #     type=str,
    #     help=(
    #         "try automatically resume from logdir//{best,last}_full.pth "
    #         "if --resume is empty"
    #     ),
    #     required=False,
    #     choices=["best", "last"],
    #     default=None,
    # )
    # parser.add_argument("--seed", type=int, default=42)
    # boolean_flag(
    #     parser,
    #     "apex",
    #     default=os.getenv("USE_APEX", "0") == "1",
    #     help="Enable/disable using of Apex extension",
    # )
    # boolean_flag(
    #     parser,
    #     "amp",
    #     default=os.getenv("USE_AMP", "0") == "1",
    #     help="Enable/disable using of PyTorch AMP extension",
    # )
    # boolean_flag(
    #     parser,
    #     "distributed",
    #     shorthand="ddp",
    #     default=os.getenv("USE_DDP", "0") == "1",
    #     help="Run in distributed mode",
    # )
    boolean_flag(parser, "verbose", default=None)
    boolean_flag(parser, "timeit", default=None)
    # boolean_flag(parser, "check", default=None)
    # boolean_flag(parser, "overfit", default=None)
    boolean_flag(
        parser,
        "deterministic",
        default=None,
        help="Deterministic mode if running in CuDNN backend",
    )
    boolean_flag(parser, "benchmark", default=None, help="Use CuDNN benchmark")

    parser.add_argument("--storage", type=int, default=None)
    parser.add_argument("--study-name", type=str, default=None)

    parser.add_argument("--direction", type=str, default=None)
    parser.add_argument("--n-trials", type=int, default=None)
    parser.add_argument("--timeout", type=int, default=None)
    parser.add_argument("--n-jobs", type=int, default=None)
    boolean_flag(parser, "gc-after-trial", default=False)
    boolean_flag(parser, "show-progress-bar", default=False)

    return parser


def parse_args():
    """Parses the command line arguments and returns arguments and config."""
    parser = argparse.ArgumentParser()
    build_args(parser)
    args, unknown_args = parser.parse_known_args()
    return args, unknown_args


def _process_trial_config(trial, config: Dict) -> Tuple[optuna.Trial, Dict]:
    def _eval_trial_suggestions(x):
        nonlocal trial
        if isinstance(x, str) and "trial.suggest_" in x:
            x = eval(x)
        return x

    config = maybe_recursive_call(config, _eval_trial_suggestions)
    return trial, config


def main(args, unknown_args):
    """Runs the ``catalyst-dl tune`` script."""
    args, config = parse_args_uargs(args, unknown_args)
    set_global_seed(args.seed)
    prepare_cudnn(args.deterministic, args.benchmark)

    # optuna objective
    def objective(trial: optuna.trial):
        trial, trial_config = _process_trial_config(trial, config.copy())
        runner: ConfigRunner = get_config_runner(expdir=Path(args.expdir), config=trial_config)
        # @TODO: here we need better solution.
        runner._trial = trial  # noqa: WPS437

        if get_rank() <= 0:
            dump_environment(logdir=runner.logdir, config=config, configs_path=args.configs)
            dump_code(expdir=args.expdir, logdir=runner.logdir)

        runner.run()

        return trial.best_score

    # optuna study
    study_params = config.pop("study", {})

    # optuna sampler
    sampler_params = study_params.pop("sampler", {})
    optuna_sampler_type = sampler_params.pop("_target_", None)
    optuna_sampler = (
        optuna.samplers.__dict__[optuna_sampler_type](**sampler_params)
        if optuna_sampler_type is not None
        else None
    )

    # optuna pruner
    pruner_params = study_params.pop("pruner", {})
    optuna_pruner_type = pruner_params.pop("_target_", None)
    optuna_pruner = (
        optuna.pruners.__dict__[optuna_pruner_type](**pruner_params)
        if optuna_pruner_type is not None
        else None
    )

    study = optuna.create_study(
        direction=args.direction or study_params.pop("direction", "minimize"),
        storage=args.storage or study_params.pop("storage", None),
        study_name=args.study_name or study_params.pop("study_name", None),
        sampler=optuna_sampler,
        pruner=optuna_pruner,
        **study_params,
    )
    study.optimize(
        objective,
        n_trials=args.n_trials,
        timeout=args.timeout,
        n_jobs=args.n_jobs or 1,
        gc_after_trial=args.gc_after_trial,
        show_progress_bar=args.show_progress_bar,
    )


if __name__ == "__main__":
    args, unknown_args = parse_args()
    main(args, unknown_args)
