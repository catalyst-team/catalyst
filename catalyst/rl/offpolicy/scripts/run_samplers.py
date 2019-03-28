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
from catalyst.rl.registry import ALGORITHMS, ENVIRONMENTS
from catalyst.utils.config import parse_args_uargs
from catalyst.utils.misc import set_global_seeds, boolean_flag
from catalyst.rl.offpolicy.sampler import Sampler
import catalyst.rl.random_process as rp

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
    parser.add_argument(
        "--action-noise-prob",
        type=float,
        default=None)
    parser.add_argument(
        "--param-noise-prob",
        type=float,
        default=None)
    parser.add_argument(
        "--max-noise-power",
        type=float,
        default=None)
    parser.add_argument(
        "--max-action-noise",
        type=float,
        default=None)
    parser.add_argument(
        "--max-param-noise",
        type=float,
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
    action_noise_prob,
    param_noise_prob,
    action_noise=None,
    param_noise=None,
    id=None,
    resume=None,
    redis=True
):
    config_ = copy.deepcopy(config)
    action_noise = action_noise or 0
    param_noise = param_noise or 0

    if not redis:
        redis_server = None
        redis_prefix = None
    else:
        redis_server = StrictRedis(
            port=config_.get("redis", {}).get("port", 12000))
        redis_prefix = config_.get("redis", {}).get("prefix", "")

    id = id or 0
    set_global_seeds(42 + id)

    if "randomized_start" in config_["env"]:
        config_["env"]["randomized_start"] = (
            config_["env"]["randomized_start"] and not infer)
    env = environment(**config_["env"], visualize=vis)
    # @TODO: remove this hack
    config_["shared"]["observation_size"] = env.observation_shape[0]
    config_["shared"]["action_size"] = env.action_shape[0]

    algo_kwargs = algorithm.prepare_for_sampler(config_)

    rp_params = config_.get("random_process", {})
    random_process = rp.__dict__[
        rp_params.pop("random_process", "RandomProcess")]
    rp_params["sigma"] = action_noise
    rp_params["size"] = config_["shared"]["action_size"]
    random_process = random_process(**rp_params)

    seeds = config_.get("seeds", None) \
        if infer \
        else config_.get("train_seeds", None)
    min_episode_steps = config_["sampler"].pop("min_episode_steps", None)
    min_episode_steps = min_episode_steps if not infer else None
    min_episode_reward = config_["sampler"].pop("min_episode_reward", None)
    min_episode_reward = min_episode_reward if not infer else None

    if seeds is not None:
        min_episode_steps = None
        min_episode_reward = None

    pprint(config_["sampler"])
    pprint(algo_kwargs)

    sampler = Sampler(
        **config_["sampler"],
        **algo_kwargs,
        env=env,
        logdir=logdir, id=id,
        redis_server=redis_server,
        redis_prefix=redis_prefix,
        mode="infer" if infer else "train",
        random_process=random_process,
        action_noise_prob=action_noise_prob,
        param_noise_prob=param_noise_prob,
        param_noise_d=param_noise,
        seeds=seeds,
        min_episode_steps=min_episode_steps,
        min_episode_reward=min_episode_reward,
        resume=resume)

    pprint(sampler)

    sampler.run()


def main(args, unknown_args):
    args, config = parse_args_uargs(args, unknown_args)

    if args.expdir is not None:
        modules = prepare_modules(  # noqa: F841
            expdir=args.expdir,
            dump_dir=args.logdir)

    algorithm = ALGORITHMS.get(args.algorithm)
    environment = ENVIRONMENTS.get(args.environment)

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
            action_noise=0.5,
            param_noise=0.5,
            action_noise_prob=args.action_noise_prob,
            param_noise_prob=args.param_noise_prob,
            id=sampler_id,
        )
        run_sampler(**params, **params_)

    for i in range(args.vis):
        params_ = dict(
            vis=False,
            infer=False,
            action_noise_prob=0,
            param_noise_prob=0,
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
            action_noise_prob=0,
            param_noise_prob=0,
            id=sampler_id,
        )
        p = mp.Process(target=run_sampler, kwargs=dict(**params, **params_))
        p.start()
        processes.append(p)
        sampler_id += 1

    for i in range(1, args.train + 1):
        action_noise = args.max_action_noise * i / args.train \
            if args.max_action_noise is not None \
            else None
        param_noise = args.max_param_noise * i / args.train \
            if args.max_param_noise is not None \
            else None
        params_ = dict(
            vis=False,
            infer=False,
            action_noise=action_noise,
            param_noise=param_noise,
            action_noise_prob=args.action_noise_prob,
            param_noise_prob=args.param_noise_prob,
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
