#!/usr/bin/env python

import argparse
import os
from pprint import pprint
from redis import StrictRedis
import torch

from catalyst.utils.config import parse_args_uargs
from catalyst.utils.misc import set_global_seeds, import_module
from catalyst.rl.offpolicy.trainer import Trainer

set_global_seeds(42)
os.environ["OMP_NUM_THREADS"] = "1"
torch.set_num_threads(1)

parser = argparse.ArgumentParser()
parser.add_argument(
    "--config",
    type=str,
    required=True)
parser.add_argument(
    "--algorithm",
    type=str,
    default=None)
parser.add_argument(
    "--logdir",
    type=str,
    default=None)
args, unknown_args = parser.parse_known_args()
args, config = parse_args_uargs(args, unknown_args, dump_config=True)

algorithm_module = import_module("algo_module", args.algorithm)
algorithm_kwargs = algorithm_module.ALGORITHM.prepare_for_trainer(config)

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
