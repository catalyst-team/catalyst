#!/usr/bin/env python

import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

import copy  # noqa E402
import time  # noqa E402
import atexit  # noqa E402
import argparse  # noqa E402
import multiprocessing as mp  # noqa E402

import torch  # noqa E402
torch.set_num_threads(1)

from catalyst.dl.scripts.utils import import_module  # noqa E402
from catalyst.rl.core import Sampler  # noqa E402
from catalyst.rl.registry import \
    OFFPOLICY_ALGORITHMS, ONPOLICY_ALGORITHMS, \
    ENVIRONMENTS, DATABASES  # noqa E402
from catalyst.rl.exploration import ExplorationHandler  # noqa E402
from catalyst.rl.scripts.utils import OFFPOLICY_ALGORITHMS_NAMES, \
    ONPOLICY_ALGORITHMS_NAMES  # noqa E402
from catalyst.utils.config import parse_args_uargs  # noqa E402
from catalyst.utils.misc import set_global_seed, boolean_flag  # noqa E402


STEP_DELAY = 1


def build_args(parser):
    parser.add_argument(
        "--config",
        "--configs",
        "-C",
        nargs="+",
        help="path to config/configs",
        metavar="CONFIG_PATH",
        dest="configs",
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
    db=True,
    exploration_power=1.0,
    sync_epoch=False
):
    config_ = copy.deepcopy(config)
    id = 0 if id is None else id
    set_global_seed(seed + id)

    db_server = DATABASES.get_from_params(
        **config.get("db", {}), sync_epoch=sync_epoch
    ) if db else None

    env = environment_fn(**config_["environment"], visualize=vis)
    agent = algorithm_fn.prepare_for_sampler(env_spec=env, config=config_)

    exploration_params = config_["sampler"].pop("exploration_params", None)
    exploration_handler = ExplorationHandler(env=env, *exploration_params) \
        if exploration_params is not None \
        else None
    if exploration_handler is not None:
        exploration_handler.set_power(exploration_power)

    mode = "infer" if infer else "train"
    valid_seeds = config_["sampler"].pop("valid_seeds", None)
    seeds = valid_seeds if infer else None

    if algorithm_fn in OFFPOLICY_ALGORITHMS.values():
        weights_sync_mode = "critic" if env.discrete_actions else "actor"
    elif algorithm_fn in ONPOLICY_ALGORITHMS.values():
        weights_sync_mode = "actor"
    else:
        # @TODO: add registry for algorithms, trainers, samplers
        raise NotImplementedError()

    sampler = Sampler(
        agent=agent,
        env=env,
        db_server=db_server,
        exploration_handler=exploration_handler,
        logdir=logdir,
        id=id,
        mode=mode,
        weights_sync_mode=weights_sync_mode,
        seeds=seeds,
        **config_["sampler"],
    )

    if resume is not None:
        sampler.load_checkpoint(filepath=resume)

    sampler.run()


def main(args, unknown_args):
    args, config = parse_args_uargs(args, unknown_args)

    if args.expdir is not None:
        module = import_module(expdir=args.expdir)  # noqa: F841

    environment_name = config["environment"].pop("environment")
    environment_fn = ENVIRONMENTS.get(environment_name)

    algorithm_name = config["algorithm"].pop("algorithm")

    if algorithm_name in OFFPOLICY_ALGORITHMS_NAMES:
        ALGORITHMS = OFFPOLICY_ALGORITHMS
        sync_epoch = False
    elif algorithm_name in ONPOLICY_ALGORITHMS_NAMES:
        ALGORITHMS = ONPOLICY_ALGORITHMS
        sync_epoch = True
    else:
        raise NotImplementedError()

    algorithm_fn = ALGORITHMS.get(algorithm_name)

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
        db=args.db,
        sync_epoch=sync_epoch
    )

    if args.check:
        params_ = dict(
            vis=(args.vis is not None and args.vis > 0),
            infer=(args.infer is not None and args.infer > 0),
            id=sampler_id
        )
        run_sampler(**params, **params_)

    for i in range(args.vis):
        params_ = dict(
            vis=True,
            infer=True,
            id=sampler_id,
            exploration_power=0.0
        )
        p = mp.Process(target=run_sampler, kwargs=dict(**params, **params_))
        p.start()
        processes.append(p)
        sampler_id += 1
        time.sleep(STEP_DELAY)

    for i in range(args.infer):
        params_ = dict(
            vis=False,
            infer=True,
            id=sampler_id,
            exploration_power=0.0
        )
        p = mp.Process(target=run_sampler, kwargs=dict(**params, **params_))
        p.start()
        processes.append(p)
        sampler_id += 1
        time.sleep(STEP_DELAY)

    for i in range(1, args.train + 1):
        exploration_power = i / args.train
        params_ = dict(
            vis=False,
            infer=False,
            id=sampler_id,
            exploration_power=exploration_power
        )
        p = mp.Process(target=run_sampler, kwargs=dict(**params, **params_))
        p.start()
        processes.append(p)
        sampler_id += 1
        time.sleep(STEP_DELAY)

    for p in processes:
        p.join()


if __name__ == "__main__":
    args, unknown_args = parse_args()
    main(args, unknown_args)
