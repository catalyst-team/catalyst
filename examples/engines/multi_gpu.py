# CUDA_VISIBLE_DEVICES="0,1" python multi_gpu.py --engine=fs-pp
from argparse import ArgumentParser, RawTextHelpFormatter
from functools import partial
import os

from torch import nn, optim
from torch.utils.data import DataLoader

from catalyst import dl, SETTINGS
from catalyst.contrib.datasets import CIFAR10
from catalyst.contrib.nn import ResidualBlock
from catalyst.data import transforms

E2E = {
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

if SETTINGS.fairscale_required:
    E2E.update(
        {
            "fs-pp": partial(dl.PipelineParallelFairScaleEngine, pipe_kwargs=dict(balance=[3, 4])),
            "fs-ddp": dl.SharedDataParallelFairScaleEngine,
            "fs-ddp-amp": dl.SharedDataParallelFairScaleAMPEngine,
            "fs-fddp": dl.FullySharedDataParallelFairScaleEngine,
        }
    )

# tested with `docker pull deepspeed/deepspeed:v031_torch17_cuda11`
if SETTINGS.deepspeed_required:
    E2E.update(
        {"ds-ddp": dl.DistributedDataParallelDeepSpeedEngine,}
    )


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
    def __init__(self, logdir, engine: str):
        super().__init__()
        self._logdir = logdir
        self._engine = engine

    def get_engine(self):
        return E2E[self._engine]()

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
        transform = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
        )
        return {
            "train": DataLoader(
                CIFAR10(os.getcwd(), train=True, download=True, transform=transform), batch_size=32
            ),
            "valid": DataLoader(
                CIFAR10(os.getcwd(), train=False, download=True, transform=transform),
                batch_size=32,
            ),
        }

    def get_model(self, stage: str):
        model = self.model if self.model is not None else resnet9(in_channels=3, num_classes=10)
        return model

    def get_criterion(self, stage: str):
        return nn.CrossEntropyLoss()

    def get_optimizer(self, stage: str, model):
        return optim.Adam(model.parameters(), lr=1e-3)

    def get_scheduler(self, stage: str, optimizer):
        return None

    def get_callbacks(self, stage: str):
        return {
            "criterion": dl.CriterionCallback(
                metric_key="loss", input_key="logits", target_key="targets"
            ),
            "optimizer": dl.OptimizerCallback(metric_key="loss"),
            # "scheduler": dl.SchedulerCallback(loader_key="valid", metric_key="loss"),
            # "accuracy": dl.AccuracyCallback(
            #     input_key="logits", target_key="targets", topk_args=(1, 3, 5)
            # ),
            # "classification": dl.PrecisionRecallF1SupportCallback(
            #     input_key="logits", target_key="targets", num_classes=10
            # ),
            # "confusion_matrix": dl.ConfusionMatrixCallback(
            #     input_key="logits", target_key="targets", num_classes=10
            # ),
            "checkpoint": dl.CheckpointCallback(
                self._logdir, loader_key="valid", metric_key="loss", minimize=True, save_n_best=1
            ),
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
    args, _ = parser.parse_known_args()
    args.logdir = args.logdir or f"logs_{args.engine}".replace("-", "_")
    runner = CustomRunner(args.logdir, args.engine)
    runner.run()
