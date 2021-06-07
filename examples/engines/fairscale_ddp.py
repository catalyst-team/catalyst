# CUDA_VISIBLE_DEVICES="0,1" python fairscale_ddp.py
import os

from torch import nn, optim
from torch.utils.data import DataLoader

from catalyst import dl, utils
from catalyst.contrib.datasets import MNIST
from catalyst.data import ToTensor


class CustomRunner(dl.IRunner):
    def __init__(self, logdir):
        super().__init__()
        self._logdir = logdir

    def get_engine(self):
        # return dl.SharedDataParallelFairScaleEngine()
        # return dl.SharedDataParallelFairScaleAMPEngine()
        return dl.FullySharedDataParallelFairScaleEngine()

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
        return {
            "train": DataLoader(
                MNIST(os.getcwd(), train=True, download=True, transform=ToTensor()), batch_size=32
            ),
            "valid": DataLoader(
                MNIST(os.getcwd(), train=False, download=True, transform=ToTensor()), batch_size=32
            ),
        }

    def get_model(self, stage: str):
        model = (
            self.model
            if self.model is not None
            else nn.Sequential(nn.Flatten(), nn.Linear(784, 128), nn.ReLU(), nn.Linear(128, 10))
        )
        return model

    def get_criterion(self, stage: str):
        return nn.CrossEntropyLoss()

    def get_optimizer(self, stage: str, model):
        # TODO: move OSS to SharedDataParallelFairScaleEngine
        from fairscale.optim import OSS

        return OSS(model.parameters(), optim=optim.Adam, lr=1e-3)
        # return optim.Adam(model.parameters(), lr=1e-3)

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
    runner = CustomRunner("./logs")
    runner.run()
