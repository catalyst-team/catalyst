from typing import Any, Callable, Dict, Optional

import numpy as np

import torch
from torch.utils.data import DataLoader

from catalyst.engines.torch import DeviceEngine
from catalyst.settings import SETTINGS

if SETTINGS.xla_required:
    import torch_xla.core.xla_model as xm
    from torch_xla.distributed.parallel_loader import ParallelLoader
    import torch_xla.distributed.xla_multiprocessing as xmp


class XLAEngine(DeviceEngine):
    """XLA SingleTPU training device engine.

    Examples:

    .. code-block:: python

        import os
        from datetime import datetime

        import torch
        from torch import nn, optim
        from torch.utils.data import DataLoader

        from catalyst import dl
        from catalyst.contrib.datasets import CIFAR10
        from catalyst.contrib.nn import ResidualBlock
        from catalyst.data import transforms

        def conv_block(in_channels, out_channels, pool=False):
            layers = [
                nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
            ]
            if pool:
                layers.append(nn.MaxPool2d(2))
            return nn.Sequential(*layers)


        def resnet9(in_channels: int, num_classes: int, size: int = 16):
            sz, sz2, sz4, sz8 = size, size * 2, size * 4, size * 8
            return nn.Sequential(
                conv_block(in_channels, sz),
                conv_block(sz, sz2, pool=True),
                ResidualBlock(nn.Sequential(conv_block(sz2, sz2), conv_block(sz2, sz2))),
                conv_block(sz2, sz4, pool=True),
                conv_block(sz4, sz8, pool=True),
                ResidualBlock(nn.Sequential(conv_block(sz8, sz8), conv_block(sz8, sz8))),
                nn.Sequential(
                    nn.MaxPool2d(4), nn.Flatten(), nn.Dropout(0.2), nn.Linear(sz8, num_classes)
                ),
            )

        class CustomRunner(dl.IRunner):
            def __init__(self, logdir):
                super().__init__()
                self._logdir = logdir

            def get_engine(self):
                return dl.XLAEngine()

            def get_loggers(self):
                return {
                    "console": dl.ConsoleLogger(),
                    "csv": dl.CSVLogger(logdir=self._logdir),
                    "tensorboard": dl.TensorboardLogger(logdir=self._logdir),
                }

            @property
            def stages(self):
                return ["train"]

            def get_stage_len(self, stage: str) -> int:
                return 3

            def get_loaders(self, stage: str):
                transform = transforms.Compose(
                    [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
                )
                train_data = CIFAR10(os.getcwd(), train=False, download=True, transform=transform)
                valid_data = CIFAR10(os.getcwd(), train=False, download=True, transform=transform)

                if self.engine.is_ddp:
                    train_sampler = torch.utils.data.distributed.DistributedSampler(
                        train_data,
                        num_replicas=self.engine.world_size,
                        rank=self.engine.rank,
                        shuffle=True
                    )
                    valid_sampler = torch.utils.data.distributed.DistributedSampler(
                        valid_data,
                        num_replicas=self.engine.world_size,
                        rank=self.engine.rank,
                        shuffle=False
                    )
                else:
                    train_sampler = valid_sampler = None

                return {
                    "train": DataLoader(train_data, batch_size=32, sampler=train_sampler),
                    "valid": DataLoader(valid_data, batch_size=32, sampler=valid_sampler),
                }

            def get_model(self, stage: str):
                model = self.model \
                    if self.model is not None \
                    else resnet9(in_channels=3, num_classes=10)
                return model

            def get_criterion(self, stage: str):
                return nn.CrossEntropyLoss()

            def get_optimizer(self, stage: str, model):
                return optim.Adam(model.parameters(), lr=1e-3)

            def get_scheduler(self, stage: str, optimizer):
                return optim.lr_scheduler.MultiStepLR(optimizer, [5, 8], gamma=0.3)

            def get_callbacks(self, stage: str):
                return {
                    "criterion": dl.CriterionCallback(
                        metric_key="loss", input_key="logits", target_key="targets"
                    ),
                    "optimizer": dl.OptimizerCallback(metric_key="loss"),
                    "scheduler": dl.SchedulerCallback(loader_key="valid", metric_key="loss"),
                    "accuracy": dl.AccuracyCallback(
                        input_key="logits", target_key="targets", topk_args=(1, 3, 5)
                    ),
                    "checkpoint": dl.CheckpointCallback(
                        self._logdir,
                        loader_key="valid",
                        metric_key="accuracy",
                        minimize=False,
                        save_n_best=1,
                    ),
                    "tqdm": dl.TqdmCallback(),
                }

            def handle_batch(self, batch):
                x, y = batch
                logits = self.model(x)

                self.batch = {
                    "features": x,
                    "targets": y,
                    "logits": logits,
                }

        logdir = f"logs/{datetime.now().strftime('%Y%m%d-%H%M%S')}"
        runner = CustomRunner(logdir)
        runner.run()
    """

    def __init__(self):
        """Init."""
        super().__init__()
        self._device = xm.xla_device()

    def optimizer_step(self, loss, model, optimizer) -> None:
        """Abstraction over ``optimizer.step()`` step."""
        xm.optimizer_step(optimizer, barrier=True)


class DistributedXLAEngine(DeviceEngine):
    """Distributed XLA MultiTPU training device engine.

    Examples:

    .. code-block:: python

        import os
        from datetime import datetime

        import torch
        from torch import nn, optim
        from torch.utils.data import DataLoader

        from catalyst import dl
        from catalyst.contrib.datasets import CIFAR10
        from catalyst.contrib.nn import ResidualBlock
        from catalyst.data import transforms

        def conv_block(in_channels, out_channels, pool=False):
            layers = [
                nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
            ]
            if pool:
                layers.append(nn.MaxPool2d(2))
            return nn.Sequential(*layers)


        def resnet9(in_channels: int, num_classes: int, size: int = 16):
            sz, sz2, sz4, sz8 = size, size * 2, size * 4, size * 8
            return nn.Sequential(
                conv_block(in_channels, sz),
                conv_block(sz, sz2, pool=True),
                ResidualBlock(nn.Sequential(conv_block(sz2, sz2), conv_block(sz2, sz2))),
                conv_block(sz2, sz4, pool=True),
                conv_block(sz4, sz8, pool=True),
                ResidualBlock(nn.Sequential(conv_block(sz8, sz8), conv_block(sz8, sz8))),
                nn.Sequential(
                    nn.MaxPool2d(4), nn.Flatten(), nn.Dropout(0.2), nn.Linear(sz8, num_classes)
                ),
            )

        class CustomRunner(dl.IRunner):
            def __init__(self, logdir):
                super().__init__()
                self._logdir = logdir

            def get_engine(self):
                return dl.DistributedXLAEngine()

            def get_loggers(self):
                return {
                    "console": dl.ConsoleLogger(),
                    "csv": dl.CSVLogger(logdir=self._logdir),
                    "tensorboard": dl.TensorboardLogger(logdir=self._logdir),
                }

            @property
            def stages(self):
                return ["train"]

            def get_stage_len(self, stage: str) -> int:
                return 3

            def get_loaders(self, stage: str):
                transform = transforms.Compose(
                    [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
                )
                train_data = CIFAR10(os.getcwd(), train=False, download=True, transform=transform)
                valid_data = CIFAR10(os.getcwd(), train=False, download=True, transform=transform)

                if self.engine.is_ddp:
                    train_sampler = torch.utils.data.distributed.DistributedSampler(
                        train_data,
                        num_replicas=self.engine.world_size,
                        rank=self.engine.rank,
                        shuffle=True
                    )
                    valid_sampler = torch.utils.data.distributed.DistributedSampler(
                        valid_data,
                        num_replicas=self.engine.world_size,
                        rank=self.engine.rank,
                        shuffle=False
                    )
                else:
                    train_sampler = valid_sampler = None

                return {
                    "train": DataLoader(train_data, batch_size=32, sampler=train_sampler),
                    "valid": DataLoader(valid_data, batch_size=32, sampler=valid_sampler),
                }

            def get_model(self, stage: str):
                model = self.model \
                    if self.model is not None \
                    else resnet9(in_channels=3, num_classes=10)
                return model

            def get_criterion(self, stage: str):
                return nn.CrossEntropyLoss()

            def get_optimizer(self, stage: str, model):
                return optim.Adam(model.parameters(), lr=1e-3)

            def get_scheduler(self, stage: str, optimizer):
                return optim.lr_scheduler.MultiStepLR(optimizer, [5, 8], gamma=0.3)

            def get_callbacks(self, stage: str):
                return {
                    "criterion": dl.CriterionCallback(
                        metric_key="loss", input_key="logits", target_key="targets"
                    ),
                    "optimizer": dl.OptimizerCallback(metric_key="loss"),
                    "scheduler": dl.SchedulerCallback(loader_key="valid", metric_key="loss"),
                    "accuracy": dl.AccuracyCallback(
                        input_key="logits", target_key="targets", topk_args=(1, 3, 5)
                    ),
                    "checkpoint": dl.CheckpointCallback(
                        self._logdir,
                        loader_key="valid",
                        metric_key="accuracy",
                        minimize=False,
                        save_n_best=1,
                    ),
                    "tqdm": dl.TqdmCallback(),
                }

            def handle_batch(self, batch):
                x, y = batch
                logits = self.model(x)

                self.batch = {
                    "features": x,
                    "targets": y,
                    "logits": logits,
                }

        logdir = f"logs/{datetime.now().strftime('%Y%m%d-%H%M%S')}"
        runner = CustomRunner(logdir)
        runner.run()
    """

    def __init__(self):
        """Init."""
        super().__init__()
        self._device = None
        self._rank = 0
        self._world_size = 8
        self._backend = "xla"

    @property
    def rank(self) -> int:
        """Process rank for distributed training."""
        return self._rank

    @property
    def world_size(self) -> int:
        """Process world size  for distributed training."""
        return self._world_size

    @property
    def backend(self) -> Optional[str]:
        """String identifier for distributed backend."""
        return self._backend

    def barrier(self) -> None:
        """
        Synchronizes all processes.

        This collective blocks processes until the all runs enter the function.
        """
        xm.rendezvous("barrier")

    def spawn(self, fn: Callable, *args: Any, **kwargs: Any) -> None:
        """Spawns abstraction for``nprocs`` creation with specified ``fn`` and ``args``/``kwargs``.

        Args:
            fn (function): Function is called as the entrypoint of the
                spawned process. This function must be defined at the top
                level of a module so it can be pickled and spawned. This
                is a requirement imposed by multiprocessing.
                The function is called as ``fn(i, *args)``, where ``i`` is
                the process index and ``args`` is the passed through tuple
                of arguments.
            *args: Arguments passed to spawn method.
            **kwargs: Keyword-arguments passed to spawn method.

        Returns:
            wrapped function.
        """
        return xmp.spawn(
            fn, args=(self._world_size,), nprocs=self._world_size, start_method="fork"
        )

    def setup_process(self, rank: int = -1, world_size: int = 1):
        """Initialize DDP variables and processes.

        Args:
            rank: process rank. Default is `-1`.
            world_size: number of devices in netwok to expect for train. Default is `1`.
        """
        self._rank = rank
        self._world_size = world_size
        self._device = xm.xla_device()

    def sync_tensor(self, tensor: torch.Tensor, mode: str) -> torch.Tensor:
        """Syncs ``tensor`` over ``world_size`` in distributed mode.

        Args:
            tensor: tensor to sync across the processes.
            mode: tensor synchronization type,
                should be one of 'sum' or 'mean'.
                Default is 'mean'.

        Returns:
            torch.Tensor with synchronized values.

        Raises:
            ValueError: if mode is out of ``sum``, ``mean``.
        """
        # return tensor
        if mode not in {"sum", "mean"}:
            raise ValueError(f"Unknown sync_type '{mode}'")
        if mode == "sum":
            return xm.all_reduce("sum", tensor)
        elif mode == "mean":
            return xm.all_reduce("sum", tensor, scale=1.0 / self.world_size)

    def sync_metrics(self, metrics: Dict) -> Dict:
        """Syncs ``metrics`` over ``world_size`` in the distributed mode."""
        metrics = {
            k: xm.mesh_reduce(k, v.item() if isinstance(v, torch.Tensor) else v, np.mean)
            for k, v in metrics.items()
        }
        return metrics

    def optimizer_step(self, loss, model, optimizer) -> None:
        """Abstraction over ``optimizer.step()`` step."""
        xm.optimizer_step(optimizer)

    def autocast_loader(self, loader: DataLoader):
        """Loader wrapper for the distributed mode."""
        return ParallelLoader(loader, [self.device]).per_device_loader(self.device)


__all__ = ["XLAEngine", "DistributedXLAEngine"]
