import os
import argparse
import pathlib

from catalyst.dl.scripts.utils import prepare_modules
from catalyst.dl.utils import UtilsFactory
from catalyst.utils.config import parse_args_uargs, save_config
from catalyst.utils.misc import set_global_seeds, boolean_flag


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--model-dir", type=str, default=None)
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--logdir", type=str, default=None)
    parser.add_argument("--baselogdir", type=str, default=None)
    parser.add_argument(
        "--resume",
        default=None,
        type=str,
        metavar="PATH",
        help="path to latest checkpoint"
    )
    parser.add_argument(
        "-j",
        "--workers",
        default=None,
        type=int,
        help="number of data loading workers"
    )
    parser.add_argument(
        "-b", "--batch-size", default=None, type=int, help="mini-batch size"
    )
    boolean_flag(parser, "verbose", default=False)

    args, unknown_args = parser.parse_known_args()
    return args, unknown_args


def main(args, unknown_args):
    args, config = parse_args_uargs(args, unknown_args, dump_config=True)
    set_global_seeds(args.seed)

    assert args.baselogdir is not None or args.logdir is not None

    if args.logdir is None:
        modules_ = prepare_modules(model_dir=args.model_dir)
        logdir = modules_["model"].prepare_logdir(config=config)
        args.logdir = str(pathlib.Path(args.baselogdir).joinpath(logdir))

    os.makedirs(args.logdir, exist_ok=True)
    save_config(config=config, logdir=args.logdir)
    modules = prepare_modules(model_dir=args.model_dir, dump_dir=args.logdir)

    model = UtilsFactory.create_model(config)
    datasource = modules["data"].DataSource()

    runner = modules["model"].ModelRunner(model=model)
    runner.train_stages(
        datasource=datasource,
        args=args,
        stages_config=config["stages"],
        verbose=args.verbose
    )


if __name__ == "__main__":
    args, unknown_args = parse_args()
    main(args, unknown_args)
