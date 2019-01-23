#!/usr/bin/env python

import os
import argparse
from pprint import pprint
from redis import StrictRedis
import torch

from catalyst.dl.scripts.utils import prepare_modules
from catalyst.contrib.registry import Registry
from catalyst.utils.config import parse_args_uargs, save_config
from catalyst.utils.misc import set_global_seeds
from catalyst.rl.offpolicy.trainer import Trainer

set_global_seeds(42)
os.environ["OMP_NUM_THREADS"] = "1"
torch.set_num_threads(1)


def build_args(parser):
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--expdir", type=str, default=None)
    parser.add_argument("--algorithm", type=str, default=None)
    parser.add_argument("--logdir", type=str, default=None)

    return parser


def parse_args():
    parser = argparse.ArgumentParser()
    build_args(parser)
    args, unknown_args = parser.parse_known_args()
    return args, unknown_args


def main(args, unknown_args):
    args, config = parse_args_uargs(args, unknown_args, dump_config=True)

    os.makedirs(args.logdir, exist_ok=True)
    save_config(config=config, logdir=args.logdir)
    if args.expdir is not None:
        modules = prepare_modules(  # noqa: F841
            expdir=args.expdir,
            dump_dir=args.logdir)

    algorithm = Registry.get_fn("algorithm", args.algorithm)
    algorithm_kwargs = algorithm.prepare_for_trainer(config)

    redis_server = StrictRedis(port=config.get("redis", {}).get("port", 12000))
    redis_prefix = config.get("redis", {}).get("prefix", "")

    pprint(config["trainer"])
    pprint(algorithm_kwargs)

    trainer = Trainer(
        **config["trainer"],
        **algorithm_kwargs,
        logdir=args.logdir,
        redis_server=redis_server,
        redis_prefix=redis_prefix)

    pprint(trainer)

    trainer.run()


if __name__ == "__main__":
    args, unknown_args = parse_args()
    main(args, unknown_args)
