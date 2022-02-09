from typing import List, Tuple
import argparse
from functools import partial

from catalyst import dl, SETTINGS

E2E = {
    "e": dl.Engine,
    "cpu": dl.CPUEngine,
    "gpu": dl.GPUEngine,
    "gpu-amp": partial(dl.GPUEngine, fp16=True),
    "dp": dl.DataParallelEngine,
    "dp-amp": partial(dl.DataParallelEngine, fp16=True),
    "ddp": dl.DistributedDataParallelEngine,
    "ddp-amp": partial(dl.DistributedDataParallelEngine, fp16=True),
}

if SETTINGS.xla_required:
    E2E.update({"xla": dl.DistributedXLAEngine})


def parse_runner_params(exp_name: str) -> Tuple[dict, List[str]]:
    """Constructs the command-line arguments for ``train_*.py``."""
    parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("--logdir", type=str, default=None)
    parser.add_argument("--engine", type=str, choices=list(E2E.keys()), default="e")
    args, unknown_args = parser.parse_known_args()
    args = vars(args)

    default_logdir = f"logs_{exp_name}_{args['engine']}".replace("-", "_")
    args["logdir"] = args["logdir"] or default_logdir

    return args, unknown_args


def parse_ddp_params(args: List[str]) -> Tuple[dict, List[str]]:
    """Constructs the command-line arguments for ``train_*.py --engine=ddp``."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--master-addr",
        default="127.0.0.1",
        type=str,
        help=(
            "Master node (rank 0)'s address,"
            " should be either the IP address or the hostname"
            " of node 0, for single node multi-proc training, can simply be '127.0.0.1'."
        ),
        dest="address",
    )
    parser.add_argument(
        "--master-port",
        default=2112,
        type=int,
        help=(
            "Master node (rank 0)'s free port that needs to be used for communication"
            " during distributed training."
        ),
        dest="port",
    )
    parser.add_argument(
        "--world-size",
        default=None,
        type=int,
        help=(
            "the number of processes to use for distributed training."
            " Should be less or equal to the number of GPUs."
        ),
        dest="world_size",
    )
    parser.add_argument(
        "--dist-rank",
        default=0,
        type=int,
        help=(
            "The rank of the first process to run on the node."
            " It should be a number between 'number of initialized processes'"
            " and 'world_size - 1'."
        ),
        dest="workers_dist_rank",
    )
    parser.add_argument(
        "--num-workers",
        default=None,
        type=int,
        help=(
            "The number of processes to launch on the node."
            " For GPU training, this is recommended to be set to the number of GPUs"
            " on the current node so that each process can be bound to a single GPU."
        ),
        dest="num_node_workers",
    )

    args, unknown_args = parser.parse_known_args(args)

    return vars(args), unknown_args


def parse_params(exp_name: str) -> Tuple[dict, List[str]]:
    """Constructs the command-line arguments for ``train_*.py``."""
    args, unknown_args = parse_runner_params(exp_name)
    if "ddp" in args["engine"]:
        ddp_args, unknown_args = parse_ddp_params(unknown_args)
        args = {**args, **ddp_args}

    return args, unknown_args
