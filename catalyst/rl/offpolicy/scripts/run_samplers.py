#!/usr/bin/env python

import os
import copy
import atexit
import argparse
from pprint import pprint
import multiprocessing as mp
from redis import StrictRedis
import torch

from catalyst.dl.scripts.utils import prepare_modules
from catalyst.contrib.registry import Registry
from catalyst.utils.config import parse_args_uargs, save_config
from catalyst.utils.misc import set_global_seeds, boolean_flag
from catalyst.rl.offpolicy.sampler import Sampler
from catalyst.rl.offpolicy.exploration import Explorator

set_global_seeds(42)
os.environ["OMP_NUM_THREADS"] = "1"
torch.set_num_threads(1)


def build_args(parser):
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--expdir", type=str, default=None)
    parser.add_argument("--algorithm", type=str, default=None)
    parser.add_argument("--environment", type=str, default=None)
    parser.add_argument("--logdir", type=str, default=None)
    parser.add_argument("--resume", type=str, default=None)

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

    boolean_flag(parser, "debug", default=False)
    boolean_flag(parser, "redis", default=True)

    return parser


def parse_args():
    parser = argparse.ArgumentParser()
    build_args(parser)
    args, unknown_args = parser.parse_known_args()
    return args, unknown_args


def run_sampler(
    *,
    logdir,
    algorithm,
    environment,
    config, vis, infer,
    id=None,
    resume=None,
    redis=True
):
    config_ = copy.deepcopy(config)

    if not redis:
        redis_server = None
        redis_prefix = None
    else:
        redis_server = StrictRedis(
            port=config_.get("redis", {}).get("port", 12000))
        redis_prefix = config_.get("redis", {}).get("prefix", "")

    id = id or 0
    set_global_seeds(42 + id)

    explorator = Explorator(config_)

    env = environment(**config_["env"], visualize=vis)
    # @TODO: remove this hack
    config_["shared"]["observation_size"] = env.observation_shape[0]
    config_["shared"]["action_size"] = env.action_shape[0]

    algo_kwargs = algorithm.prepare_for_sampler(config_)

    seeds = config_.get("seeds", None) \
        if infer \
        else config_.get("train_seeds", None)

    pprint(config_["sampler"])
    pprint(algo_kwargs)

    sampler = Sampler(
        **config_["sampler"],
        **algo_kwargs,
        explorator=explorator,
        env=env,
        logdir=logdir, id=id,
        redis_server=redis_server,
        redis_prefix=redis_prefix,
        mode="infer" if infer else "train",
        seeds=seeds,
        resume=resume,
        discrete_actions=config_["shared"]["discrete_actions"]
    )

    pprint(sampler)

    sampler.run()


def main(args, unknown_args):
    args, config = parse_args_uargs(args, unknown_args)

    os.makedirs(args.logdir, exist_ok=True)
    save_config(config=config, logdir=args.logdir)
    if args.expdir is not None:
        modules = prepare_modules(  # noqa: F841
            expdir=args.expdir,
            dump_dir=args.logdir)

    algorithm = Registry.get_fn("algorithm", args.algorithm)
    environment = Registry.get_fn("environment", args.environment)

    processes = []
    sampler_id = 0

    def on_exit():
        for p in processes:
            p.terminate()

    atexit.register(on_exit)

    params = dict(
        logdir=args.logdir,
        algorithm=algorithm,
        environment=environment,
        config=config,
        resume=args.resume,
        redis=args.redis
    )

    if args.debug:
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
