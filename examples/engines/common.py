from typing import List, Tuple
import argparse
from functools import partial

from catalyst import dl, SETTINGS, utils

E2E = {
    "de": dl.DeviceEngine,
    "dp": dl.DataParallelEngine,
    "ddp": dl.DistributedDataParallelEngine,
}

if SETTINGS.amp_required:
    E2E.update(
        {"amp-dp": dl.DataParallelAMPEngine, "amp-ddp": dl.DistributedDataParallelAMPEngine}
    )

if SETTINGS.apex_required:
    E2E.update(
        {"apex-dp": dl.DataParallelAPEXEngine, "apex-ddp": dl.DistributedDataParallelAPEXEngine}
    )

if SETTINGS.deepspeed_required:
    E2E.update({"ds-ddp": dl.DistributedDataParallelDeepSpeedEngine})

if SETTINGS.fairscale_required:
    E2E.update(
        {
            "fs-pp": dl.PipelineParallelFairScaleEngine,
            "fs-ddp": dl.SharedDataParallelFairScaleEngine,
            "fs-ddp-amp": dl.SharedDataParallelFairScaleAMPEngine,
            # for some reason we could catch a bug with FairScale flatten wrapper here, so...
            "fs-fddp": partial(
                dl.FullySharedDataParallelFairScaleEngine, ddp_kwargs={"flatten_parameters": False}
            ),
        }
    )

if SETTINGS.xla_required:
    E2E.update({"xla": dl.XLAEngine, "xla-ddp": dl.DistributedXLAEngine})


def parse_ddp_params(args: List[str]) -> Tuple[dict, List[str]]:
    """Constructs the command-line arguments for ``train_*.py --engine=ddp``."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--master-addr",
        default="127.0.0.1",
        type=str,
        help=(
            "Master node (rank 0)'s address, should be either the IP address or the hostname"
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
        "--world_size",
        default=None,
        type=int,
        help=(
            "the number of processes to use for distributed training."
            " Should be less or equal to the number of GPUs."
        ),
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

    # TODO: add `process_group_kwargs`
    utils.boolean_flag(
        parser,
        "sync-bn",
        default=False,
        help="Batchnorm synchonization during disributed training.",
    )
    # TODO: add `ddp_kwargs`

    args, unknown_args = parser.parse_known_args(args)

    return vars(args), unknown_args
