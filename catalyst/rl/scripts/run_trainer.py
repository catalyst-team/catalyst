#!/usr/bin/env python

import os
import argparse

from catalyst.utils.scripts import import_module
from catalyst.utils import parse_args_uargs, boolean_flag, \
    dump_environment, set_global_seed, prepare_cudnn
from catalyst.utils.scripts import dump_code
from catalyst.rl.registry import OFFPOLICY_ALGORITHMS, ONPOLICY_ALGORITHMS, \
    ENVIRONMENTS, DATABASES
from catalyst.rl.offpolicy.trainer import Trainer as OffpolicyTrainer
from catalyst.rl.onpolicy.trainer import Trainer as OnpolicyTrainer
from catalyst.rl.scripts.misc import OFFPOLICY_ALGORITHMS_NAMES, \
    ONPOLICY_ALGORITHMS_NAMES


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
    # parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--seed", type=int, default=42)

    boolean_flag(
        parser, "deterministic",
        default=None,
        help="Deterministic mode if running in CuDNN backend"
    )
    boolean_flag(
        parser, "benchmark",
        default=None,
        help="Use CuDNN benchmark"
    )

    return parser


def parse_args():
    parser = argparse.ArgumentParser()
    build_args(parser)
    args, unknown_args = parser.parse_known_args()
    return args, unknown_args


def main(args, unknown_args):
    args, config = parse_args_uargs(args, unknown_args)
    set_global_seed(args.seed)
    prepare_cudnn(args.deterministic, args.benchmark)

    if args.logdir is not None:
        os.makedirs(args.logdir, exist_ok=True)
        dump_environment(config, args.logdir, args.configs)

    if args.expdir is not None:
        module = import_module(expdir=args.expdir)  # noqa: F841
        if args.logdir is not None:
            dump_code(args.expdir, args.logdir)

    env = ENVIRONMENTS.get_from_params(**config["environment"])

    algorithm_name = config["algorithm"].pop("algorithm")
    if algorithm_name in OFFPOLICY_ALGORITHMS_NAMES:
        ALGORITHMS = OFFPOLICY_ALGORITHMS
        trainer_fn = OffpolicyTrainer
        sync_epoch = False
    elif algorithm_name in ONPOLICY_ALGORITHMS_NAMES:
        ALGORITHMS = ONPOLICY_ALGORITHMS
        trainer_fn = OnpolicyTrainer
        sync_epoch = True
    else:
        # @TODO: add registry for algorithms, trainers, samplers
        raise NotImplementedError()

    db_server = DATABASES.get_from_params(
        **config.get("db", {}), sync_epoch=sync_epoch
    )

    algorithm_fn = ALGORITHMS.get(algorithm_name)
    algorithm = algorithm_fn.prepare_for_trainer(env_spec=env, config=config)

    # if args.resume is not None:
    #     algorithm.load_checkpoint(filepath=args.resume)

    monitoring_params = config.get("monitoring_params", None)

    trainer = trainer_fn(
        algorithm=algorithm,
        env_spec=env,
        db_server=db_server,
        logdir=args.logdir,
        monitoring_params=monitoring_params,
        **config["trainer"],
    )

    trainer.run()


if __name__ == "__main__":
    args, unknown_args = parse_args()
    main(args, unknown_args)
