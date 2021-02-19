# flake8: noqa

from typing import Any, Dict, List
import logging
from tempfile import TemporaryDirectory

from pytest import mark
import torch
from torch.utils.data import DataLoader

from catalyst import dl
from catalyst.engines.device import DeviceEngine
from catalyst.settings import IS_CUDA_AVAILABLE, NUM_CUDA_DEVICES

from .utils import DummyModel, DummyDataset, DeviceCheckCallback, LossMinimizationCallback

logger = logging.getLogger(__name__)


# experiment definition
class CustomRunner(dl.IRunner):
    def __init__(self, logdir, device):
        super().__init__()
        self._logdir = logdir
        self._device = device

    def get_engine(self):
        return DeviceEngine(self._device)

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


def run_train_with_experiment_device(device):
    with TemporaryDirectory() as logdir:
        runner = CustomRunner(logdir, device)
        runner.run()


def run_train_with_config_experiment_device(device):
    pass
    # dataset = DummyDataset(10)
    # runner = SupervisedRunner()
    # logdir = f"./test_{device}_engine"
    # exp = ConfigExperiment(
    #     config={
    #         "model_params": {"_target_": "DummyModel", "in_features": 4, "out_features": 1,},
    #         "engine": str(device),
    #         "args": {"logdir": logdir},
    #         "stages": {
    #             "data_params": {"batch_size": 4, "num_workers": 0},
    #             "criterion_params": {"_target_": "MSELoss"},
    #             "optimizer_params": {"_target_": "SGD", "lr": 1e-3},
    #             "stage1": {
    #                 "stage_params": {"num_epochs": 2},
    #                 "callbacks_params": {
    #                     "loss": {"_target_": "CriterionCallback"},
    #                     "optimizer": {"_target_": "OptimizerCallback"},
    #                     "test_device": {
    #                         "_target_": "DeviceCheckCallback",
    #                         "assert_device": str(device),
    #                     },
    #                     "test_loss_minimization": {"_target_": "LossMinimizationCallback",},
    #                 },
    #             },
    #         },
    #     }
    # )
    # exp.get_datasets = lambda *args, **kwargs: {
    #     "train": dataset,
    #     "valid": dataset,
    # }
    # runner.run(exp)
    # shutil.rmtree(logdir, ignore_errors=True)


def test_experiment_engine_with_cpu():
    run_train_with_experiment_device("cpu")


@mark.skip("Config experiment is in development phase!")
def test_config_experiment_engine_with_cpu():
    run_train_with_config_experiment_device("cpu")


@mark.skipif(not IS_CUDA_AVAILABLE, reason="CUDA device is not available")
def test_experiment_engine_with_cuda():
    run_train_with_experiment_device("cuda:0")


@mark.skip("Config experiment is in development phase!")
@mark.skipif(not IS_CUDA_AVAILABLE, reason="CUDA device is not available")
def test_config_experiment_engine_with_cuda():
    run_train_with_config_experiment_device("cuda:0")


@mark.skipif(
    not IS_CUDA_AVAILABLE and NUM_CUDA_DEVICES < 2, reason="Number of CUDA devices is less than 2",
)
def test_experiment_engine_with_another_cuda_device():
    run_train_with_experiment_device("cuda:1")


@mark.skip("Config experiment is in development phase!")
@mark.skipif(
    not IS_CUDA_AVAILABLE and NUM_CUDA_DEVICES < 2, reason="Number of CUDA devices is less than 2",
)
def test_config_experiment_engine_with_another_cuda_device():
    run_train_with_config_experiment_device("cuda:1")
