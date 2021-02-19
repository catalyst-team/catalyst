# flake8: noqa

from typing import Any, Dict, List
import logging
from tempfile import TemporaryDirectory

from pytest import mark
import torch
from torch.utils.data import DataLoader

from catalyst import dl
from catalyst.engines.amp import AMPEngine
from catalyst.settings import IS_CUDA_AVAILABLE, NUM_CUDA_DEVICES

from .utils import DummyDataset, DummyModel, LossMinimizationCallback, TensorTypeChecker, DeviceCheckCallback

logger = logging.getLogger(__name__)


class CustomRunner(dl.IRunner):
    def __init__(self, logdir, device):
        super().__init__()
        self._logdir = logdir
        self._device = device

    def get_engine(self):
        return AMPEngine(self._device)

    def get_callbacks(self, stage: str) -> Dict[str, dl.Callback]:
        return {
            "criterion": dl.CriterionCallback(metric_key="loss", input_key="logits", target_key="targets"),
            "optimizer": dl.OptimizerCallback(metric_key="loss"),
            # "scheduler": dl.SchedulerCallback(loader_key="valid", metric_key="loss"),
            "checkpoint": dl.CheckpointCallback(
                self._logdir, loader_key="valid", metric_key="loss", minimize=True, save_n_best=3
            ),
            "check": DeviceCheckCallback(self._device),
            "check2": LossMinimizationCallback("loss"),
            "logits_type_checker": TensorTypeChecker("logits"),
            # "loss_type_checker": TensorTypeChecker("loss", True),
        }

    @property
    def stages(self) -> "Iterable[str]":
        return ["train"]

    def get_stage_len(self, stage: str) -> int:
        return 3

    def get_loaders(self, stage: str) -> "OrderedDict[str, DataLoader]":
        dataset = DummyDataset(6)
        loader = DataLoader(dataset, batch_size=4)
        return {"train": loader, "valid": loader}

    def get_model(self, stage: str):
        return DummyModel(4, 2)

    def get_criterion(self, stage: str):
        return torch.nn.MSELoss()

    def get_optimizer(self, model, stage: str):
        return torch.optim.Adam(model.parameters())

    # TODO: fix this
    def _get_optimizer(self, *args, **kwargs):
        assert self.model is not None, "You need to setup model first"
        self.optimizer = self.get_optimizer(stage=self.stage_key, model=self.model)
        return self.optimizer

    def get_scheduler(self, optimizer, stage: str):
        return None

    # TODO: fix this
    def _get_scheduler(self, *args, **kwargs):
        assert self.optimizer is not None, "You need to setup optimizer first"
        self.scheduler = self.get_scheduler(stage=self.stage_key, optimizer=self.optimizer)
        return self.scheduler

    def get_trial(self):
        return None

    def get_loggers(self):
        return {"console": dl.ConsoleLogger(), "csv": dl.CSVLogger(logdir=self._logdir)}

    def handle_batch(self, batch):
        x, y = batch
        logits = self.model(x)

        self.batch = {"features": x, "targets": y, "logits": logits}


def run_train_with_experiment_amp_device(device):
    # dataset = DummyDataset(10)
    # loader = DataLoader(dataset, batch_size=4)
    # runner = SupervisedRunner()
    # exp = Experiment(
    #     model=_model_fn,
    #     criterion=nn.MSELoss(),
    #     optimizer=_optimizer_fn,
    #     loaders={"train": loader, "valid": loader},
    #     main_metric="loss",
    #     callbacks=[
    #         CriterionCallback(),
    #         OptimizerCallback(),
    #         # DeviceCheckCallback(device),
    #         LossMinimizationCallback(),
    #     ],
    #     engine=DataParallelEngine(),
    # )
    with TemporaryDirectory() as logdir:
        runner = CustomRunner(logdir, device)
        runner.run()


@mark.skipif(not IS_CUDA_AVAILABLE, reason="CUDA device is not available")
def test_experiment_engine_with_cuda():
    run_train_with_experiment_amp_device("cuda:0")


@mark.skip("Config experiment is in development phase!")
@mark.skipif(not IS_CUDA_AVAILABLE, reason="CUDA device is not available")
def test_config_experiment_engine_with_cuda():
    run_train_with_experiment_amp_device("cuda:0")


@mark.skipif(
    not IS_CUDA_AVAILABLE and NUM_CUDA_DEVICES < 2, reason="Number of CUDA devices is less than 2",
)
def test_experiment_engine_with_another_cuda_device():
    run_train_with_experiment_amp_device("cuda:1")


@mark.skip("Config experiment is in development phase!")
@mark.skipif(
    not IS_CUDA_AVAILABLE and NUM_CUDA_DEVICES < 2, reason="Number of CUDA devices is less than 2",
)
def test_config_experiment_engine_with_another_cuda_device():
    run_train_with_experiment_amp_device("cuda:1")
