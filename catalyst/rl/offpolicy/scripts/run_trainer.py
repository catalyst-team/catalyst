#!/usr/bin/env python

import os
import atexit
import argparse
from redis import StrictRedis
import torch

from catalyst.dl.scripts.utils import import_module
from catalyst.contrib.registry import Registry
from catalyst.utils.config import parse_args_uargs
from catalyst.utils.misc import set_global_seed
from catalyst.rl.offpolicy.trainer import Trainer

os.environ["OMP_NUM_THREADS"] = "1"
torch.set_num_threads(1)


def build_args(parser):
    parser.add_argument(
        "-C",
        "--config",
        help="path to config/configs",
        required=True
    )
    parser.add_argument("--expdir", type=str, default=None)
    parser.add_argument("--logdir", type=str, default=None)
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--seed", type=int, default=42)

    return parser


def parse_args():
    parser = argparse.ArgumentParser()
    build_args(parser)
    args, unknown_args = parser.parse_known_args()
    return args, unknown_args


def main(args, unknown_args):
    args, config = parse_args_uargs(args, unknown_args, dump_config=True)
    set_global_seed(args.seed)

    module = import_module(expdir=args.expdir)  # noqa: F841

    redis_server = StrictRedis(port=config.get("redis", {}).get("port", 12000))
    redis_prefix = config.get("redis", {}).get("prefix", "")

    environment_fn = Registry.get_fn("environment", args.environment)
    env = environment_fn(**config["environment"])
    config["environment"] = \
        env.update_environment_config(config["environment"])
    del env

    algorithm = Registry.get_fn("algorithm", args.algorithm)
    algorithm_kwargs = algorithm.prepare_for_trainer(config)

    trainer = Trainer(
        **config["trainer"],
        **algorithm_kwargs,
        logdir=args.logdir,
        redis_server=redis_server,
        redis_prefix=redis_prefix,
        resume=args.resume,
    )

    def on_exit():
        for p in trainer.get_processes():
            p.terminate()

    atexit.register(on_exit)

    trainer.run()


if __name__ == "__main__":
    args, unknown_args = parse_args()
    main(args, unknown_args)
