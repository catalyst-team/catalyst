#!/usr/bin/env python
# flake8: noqa
from typing import Optional
from argparse import ArgumentParser, RawTextHelpFormatter
import os

from common import E2E, parse_ddp_params

from torch import nn, optim
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from catalyst import dl, utils
from catalyst.contrib import CIFAR10, Compose, ImageToTensor, NormalizeImage, ResidualBlock


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
        nn.Sequential(nn.MaxPool2d(4), nn.Flatten(), nn.Dropout(0.2), nn.Linear(sz8, num_classes)),
    )


class CustomRunner(dl.IRunner):
    def __init__(self, logdir: str, engine: str, engine_params: Optional[dict] = None):
        super().__init__()
        self._logdir = logdir
        self._engine = engine
        self._engine_params = engine_params or {}

    def get_engine(self):
        return E2E[self._engine](**self._engine_params)

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
        return 10

    def get_loaders(self, stage: str):
        transform = Compose(
            [
                ImageToTensor(),
                NormalizeImage((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ]
        )
        train_data = CIFAR10(os.getcwd(), train=True, download=True, transform=transform)
        valid_data = CIFAR10(os.getcwd(), train=False, download=True, transform=transform)
        if self.engine.is_ddp:
            train_sampler = DistributedSampler(
                train_data,
                num_replicas=self.engine.world_size,
                rank=self.engine.rank,
                shuffle=True,
            )
            valid_sampler = DistributedSampler(
                valid_data,
                num_replicas=self.engine.world_size,
                rank=self.engine.rank,
                shuffle=False,
            )
        else:
            train_sampler = valid_sampler = None

        return {
            "train": DataLoader(train_data, batch_size=32, sampler=train_sampler, num_workers=4),
            "valid": DataLoader(valid_data, batch_size=32, sampler=valid_sampler, num_workers=4),
        }

    def get_model(self, stage: str):
        model = self.model if self.model is not None else resnet9(in_channels=3, num_classes=10)
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
                metric_key="accuracy01",
                minimize=False,
                save_n_best=1,
            ),
            # "tqdm": dl.TqdmCallback(),
        }

    def handle_batch(self, batch):
        x, y = batch
        logits = self.model(x)
        self.batch = {
            "features": x,
            "targets": y,
            "logits": logits,
        }


if __name__ == "__main__":
    parser = ArgumentParser(formatter_class=RawTextHelpFormatter)
    parser.add_argument("--logdir", type=str, default=None)
    parser.add_argument("--engine", type=str, choices=list(E2E.keys()))
    args, unknown_args = parser.parse_known_args()
    args.logdir = args.logdir or f"logs_resnet_{args.engine}".replace("-", "_")
    if args.engine in {"ddp", "amp-ddp", "apex-ddp", "ds-ddp", "fs-ddp", "fs-ddp-amp", "fs-fddp"}:
        engine_params, _ = parse_ddp_params(unknown_args)

        # fix for DeepSpeed engine since is does not support batchnorm synchonization
        if args.engine == "ds-ddp":
            engine_params.pop("sync_bn")
    else:
        engine_params = None

    runner = CustomRunner(args.logdir, args.engine, engine_params)
    runner.run()
