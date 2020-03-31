from collections import OrderedDict
import os

import torch
from torch.multiprocessing import Process
from torch.utils.data import DataLoader, DistributedSampler

from catalyst.data import DistributedSamplerWrapper


def _process_dataloader(loader):
    sampler = (
        DistributedSampler(dataset=loader.dataset)
        if loader.sampler is not None
        else DistributedSamplerWrapper(sampler=loader.sampler)
    )
    return DataLoader(
        dataset=loader.dataset.copy(),
        sampler=sampler,
        batch_size=loader.batch_size,
        num_workers=loader.num_workers,
    )


def _worker_fn(
    rank, world_size, experiment, runner,
):
    print("check")

    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = "29500"
    torch.distributed.init_process_group(
        backend="nccl", init_method="env://", rank=rank, world_size=world_size
    )

    # torch.cuda.set_device(int(rank))

    experiment.loaders = OrderedDict(
        [
            (key, _process_dataloader(value))
            for key, value in experiment.loaders.items()
        ]
    )

    runner.run_experiment(experiment)


def distributed_exp_run(experiment, runner):
    world_size = torch.cuda.device_count()
    processes = []
    try:
        for rank in range(world_size):
            p = Process(
                target=_worker_fn, args=(rank, world_size, experiment, runner)
            )
            p.start()
            processes.append(p)
        for p in processes:
            p.join()
    finally:
        for process in processes:
            process.kill()


__all__ = ["distributed_exp_run"]
