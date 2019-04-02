#!/usr/bin/env python

import os
import copy
import atexit
import argparse
import multiprocessing as mp
import torch

from catalyst.dl.scripts.utils import import_module
from catalyst.rl.registry import ALGORITHMS, ENVIRONMENTS
from catalyst.utils.config import parse_args_uargs
from catalyst.utils.misc import set_global_seed, boolean_flag
from catalyst.rl.offpolicy.sampler import Sampler
from catalyst.rl.db.redis import RedisDB

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

    parser.add_argument(
        "--vis",
        type=int,
        default=None)
    parser.add_argument(
        "--infer",
        type=int,
        default=None)
    parser.add_argument(
        "--train",
        type=int,
        default=None)

    boolean_flag(parser, "check", default=False)
    boolean_flag(parser, "db", default=True)

    return parser


def parse_args():
    parser = argparse.ArgumentParser()
    build_args(parser)
    args, unknown_args = parser.parse_known_args()
    return args, unknown_args


def run_sampler(
    *,
    config,
    logdir,
    algorithm_fn,
    environment_fn,
    vis,
    infer,
    seed=42,
    id=None,
    resume=None,
    db=True
):
    config_ = copy.deepcopy(config)
    id = 0 if id is None else id
    set_global_seed(seed + id)

    db_server = RedisDB(
        port=config.get("redis", {}).get("port", 12000),
        prefix=config.get("redis", {}).get("prefix", "")
    ) if db else None

    env = environment_fn(**config_["environment"], visualize=vis)
    agent = algorithm_fn.prepare_for_sampler(config_)

    sampler = Sampler(
        agent=agent,
        env=env,
        db_server=db_server,
        **config_["sampler"],
        logdir=logdir,
        id=id,
        mode="infer" if infer else "train",
        resume=resume
    )

    sampler.run()


def main(args, unknown_args):
    args, config = parse_args_uargs(args, unknown_args)

    if args.expdir is not None:
        module = import_module(expdir=args.expdir)  # noqa: F841

    algorithm_fn = ALGORITHMS.get(args.algorithm)
    environment_fn = ENVIRONMENTS.get(args.environment)

    processes = []
    sampler_id = 0

    def on_exit():
        for p in processes:
            p.terminate()

    atexit.register(on_exit)

    params = dict(
        seed=args.seed,
        logdir=args.logdir,
        algorithm_fn=algorithm_fn,
        environment_fn=environment_fn,
        config=config,
        resume=args.resume,
        db=args.db
    )

    if args.check:
        params_ = dict(
            vis=False,
            infer=False,
            id=sampler_id,
        )
        run_sampler(**params, **params_)

    for i in range(args.vis):
        params_ = dict(
            vis=False,
            infer=False,
            id=sampler_id,
        )
        p = mp.Process(target=run_sampler, kwargs=dict(**params, **params_))
        p.start()
        processes.append(p)
        sampler_id += 1

    for i in range(args.infer):
        params_ = dict(
            vis=False,
            infer=True,
            id=sampler_id,
        )
        p = mp.Process(target=run_sampler, kwargs=dict(**params, **params_))
        p.start()
        processes.append(p)
        sampler_id += 1

    for i in range(1, args.train + 1):
        params_ = dict(
            vis=False,
            infer=False,
            id=sampler_id,
        )
        p = mp.Process(target=run_sampler, kwargs=dict(**params, **params_))
        p.start()
        processes.append(p)
        sampler_id += 1

    for p in processes:
        p.join()


if __name__ == "__main__":
    args, unknown_args = parse_args()
    main(args, unknown_args)
