#!/usr/bin/env python
import argparse
from argparse import ArgumentParser
import os
from pathlib import Path
import sys

from catalyst.dl.scripts.misc import parse_args_uargs
from catalyst.runners.config import ConfigRunner
from catalyst.settings import SETTINGS
from catalyst.utils.distributed import get_rank
from catalyst.utils.misc import boolean_flag, set_global_seed
from catalyst.utils.sys import dump_code, dump_environment, get_config_runner
from catalyst.utils.torch import prepare_cudnn

if SETTINGS.hydra_required:
    from catalyst.dl.scripts.hydra_run import main as hydra_main


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
        required=False,
    )
    parser.add_argument("--expdir", type=str, default=None)
    parser.add_argument("--logdir", type=str, default=None)
    parser.add_argument("--baselogdir", type=str, default=None)
    # parser.add_argument(
    #     "--resume", default=None, type=str, metavar="PATH", help="path to latest checkpoint",
    # )
    # parser.add_argument(
    #     "--autoresume",
    #     type=str,
    #     help=(
    #         "try automatically resume from logdir//{best,last}_full.pth " "if --resume is empty"
    #     ),
    #     required=False,
    #     choices=["best", "last"],
    #     default=None,
    # )
    parser.add_argument("--seed", type=int, default=42)
    boolean_flag(
        parser,
        "apex",
        default=os.getenv("USE_APEX", "0") == "1",
        help="Enable/disable using of Nvidia Apex extension",
    )
    boolean_flag(
        parser,
        "amp",
        default=os.getenv("USE_AMP", "0") == "1",
        help="Enable/disable using of PyTorch AMP extension",
    )
    boolean_flag(
        parser,
        "fp16",
        default=os.getenv("USE_FP16", "0") == "1",
        help="Run in half-precision mode",
    )
    boolean_flag(
        parser, "ddp", default=os.getenv("USE_DDP", "0") == "1", help="Run in distributed mode",
    )
    boolean_flag(parser, "verbose", default=None)
    boolean_flag(parser, "timeit", default=None)
    boolean_flag(parser, "check", default=None)
    boolean_flag(parser, "overfit", default=None)
    boolean_flag(
        parser,
        "deterministic",
        default=None,
        help="Deterministic mode if running in CuDNN backend",
    )
    boolean_flag(parser, "benchmark", default=None, help="Use CuDNN benchmark")
    boolean_flag(parser, "hydra", default=None, help="Use Hydra")

    return parser


def parse_args():
    """Parses the command line arguments and returns arguments and config."""
    parser = argparse.ArgumentParser()
    build_args(parser)
    args, unknown_args = parser.parse_known_args()
    return args, unknown_args


def config_main(args, unknown_args):
    """Yaml config catalyst-dl run entry point."""
    args, config = parse_args_uargs(args, unknown_args)
    set_global_seed(args.seed)
    prepare_cudnn(args.deterministic, args.benchmark)

    runner: ConfigRunner = get_config_runner(expdir=Path(args.expdir), config=config)

    if get_rank() <= 0:
        dump_environment(logdir=runner.logdir, config=config, configs_path=args.configs)
        dump_code(expdir=args.expdir, logdir=runner.logdir)

    runner.run()


def main(args, unknown_args):
    """Runs the ``catalyst-dl run`` script."""
    if args.hydra:
        assert SETTINGS.hydra_required, (
            "catalyst[hydra] requirements are not available, to install them,"
            " run `pip install catalyst[hydra]`."
        )
    if args.hydra:
        sys.argv.remove("run")
        sys.argv.remove("--hydra")
        hydra_main()
    else:
        config_main(args, unknown_args)


if __name__ == "__main__":
    args, unknown_args = parse_args()
    main(args, unknown_args)
