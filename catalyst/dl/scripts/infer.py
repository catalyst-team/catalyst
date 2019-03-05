#!/usr/bin/env python

import argparse

from catalyst.dl.scripts.utils import prepare_modules
from catalyst.contrib.registry import Registry
from catalyst.utils.config import parse_args_uargs
from catalyst.utils.misc import set_global_seeds, boolean_flag


def build_args(parser):
    parser.add_argument("--expdir", type=str, default=None)
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--seed", type=int, default=42)
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
        metavar="N",
        help="number of data loading workers"
    )
    parser.add_argument(
        "-b",
        "--batch-size",
        default=None,
        type=int,
        metavar="N",
        help="mini-batch size"
    )
    boolean_flag(parser, "verbose", default=False)
    parser.add_argument("--out-prefix", type=str, default=None)

    return parser


def parse_args():
    parser = argparse.ArgumentParser()
    build_args(parser)
    args, unknown_args = parser.parse_known_args()
    return args, unknown_args


def main(args, unknown_args):
    args, config = parse_args_uargs(args, unknown_args)
    set_global_seeds(args.seed)

    modules = prepare_modules(expdir=args.expdir)

    model = Registry.get_model(**config["model_params"])
    datasource = modules["data"].DataSource()
    data_params = config.get("data_params", {}) or {}
    loaders = datasource.prepare_loaders(
        mode="infer",
        n_workers=args.workers,
        batch_size=args.batch_size,
        **data_params
    )

    runner = modules["model"].ModelRunner(model=model)
    callbacks_params = config.get("callbacks_params", {}) or {}
    callbacks = runner.prepare_callbacks(
        mode="infer",
        resume=args.resume,
        out_prefix=args.out_prefix,
        **callbacks_params
    )
    runner.infer(loaders=loaders, callbacks=callbacks, verbose=args.verbose)


if __name__ == "__main__":
    args, unknown_args = parse_args()
    main(args, unknown_args)
