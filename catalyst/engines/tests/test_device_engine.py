# flake8: noqa

from typing import Any, Dict, List
import logging
import shutil

from pytest import mark
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset

from catalyst import dl

# from catalyst.callbacks import CriterionCallback, OptimizerCallback
from catalyst.core.callback import Callback, CallbackOrder

# from catalyst.runners.supervised import SupervisedRunner
from catalyst.core.runner import IRunner, IStageBasedRunner
from catalyst.engines.device import DeviceEngine

# from catalyst.experiments.config import ConfigExperiment
# from catalyst.experiments.misc import SingleStageExperiment as Experiment
from catalyst.registry import REGISTRY
from catalyst.settings import IS_CUDA_AVAILABLE, NUM_CUDA_DEVICES

logger = logging.getLogger(__name__)


class DummyDataset(Dataset):
    """
    Dummy dataset.
    """

    features_dim: int = 4
    out_dim: int = 2

    def __init__(self, num_records: int):
        self.num_records = num_records

    def __len__(self):
        """
        Returns:
            dataset's length.
        """
        return self.num_records

    def __getitem__(self, idx: int):
        """
        Args:
            idx: index of sample

        Returns:
            dummy features and targets vector
        """
        x = torch.ones(self.features_dim, dtype=torch.float)
        y = torch.ones(self.out_dim, dtype=torch.float)
        return x, y


@REGISTRY.add
class DummyModel(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.layers = nn.Linear(in_features, out_features)

    def forward(self, batch):
        return self.layers(batch)


@REGISTRY.add
class DeviceCheckCallback(Callback):
    def __init__(self, assert_device: str):
        super().__init__(order=CallbackOrder.internal)
        self.device = torch.device(assert_device)

    def on_stage_start(self, runner: "IRunner"):
        model_device = next(runner.model.parameters()).device
        logger.warning(
            f"DeviceCheckCallback: model device ({model_device}) - device ({self.device})"
        )
        assert model_device == self.device


@REGISTRY.add
class LossMinimizationCallback(Callback):
    def __init__(self, key="loss"):
        super().__init__(order=CallbackOrder.metric)
        self.key = key
        self.container = None
        self.round_nums = 6

    def on_epoch_start(self, runner: "IRunner"):
        self.container = []

    def on_batch_end(self, runner: "IRunner"):
        self.container.append(runner.batch_metrics[self.key].item())

    def on_epoch_end(self, runner: "IRunner"):
        assert len(self.container)

        container = [round(num, self.round_nums) for num in self.container]
        logger.warning(f"LossMinimizationCallback: {container}")
        assert all(round(a, 6) >= round(b, 6) for a, b in zip(container[:-1], container[1:]))

        self.container = []


# experiment definition
class CustomExperiment(dl.IExperiment):
    def __init__(self, device):
        self._device = device

    @property
    def seed(self) -> int:
        return 73

    @property
    def name(self) -> str:
        return "experiment73"

    @property
    def hparams(self) -> Dict:
        return {}

    @property
    def stages(self) -> List[str]:
        return ["train"]

    def get_stage_params(self, stage: str) -> Dict[str, Any]:
        return {
            "num_epochs": 10,
            "migrate_model_from_previous_stage": False,
            "migrate_callbacks_from_previous_stage": False,
        }

    def get_loaders(self, stage: str, epoch: int = None) -> Dict[str, Any]:
        dataset = DummyDataset(6)
        loader = DataLoader(dataset, batch_size=4)
        return {"train": loader, "valid": loader}

    def get_model(self, stage: str):
        return DummyModel(4, 2)

    def get_criterion(self, stage: str):
        return nn.MSELoss()

    def get_optimizer(self, stage: str, model):
        return optim.SGD(model.parameters(), lr=1e-3)

    def get_scheduler(self, stage: str, optimizer):
        return None

    def get_callbacks(self, stage: str) -> Dict[str, dl.Callback]:
        return {
            "criterion": dl.CriterionCallback(
                metric_key="loss", input_key="logits", target_key="targets"
            ),
            "optimizer": dl.OptimizerCallback(metric_key="loss"),
            # "scheduler": dl.SchedulerCallback(loader_key="valid", metric_key="loss"),
            "checkpoint": dl.CheckpointCallback(
                loader_key="valid", metric_key="loss", minimize=True, save_n_best=3
            ),
            "check": DeviceCheckCallback(self._device),
            "check2": LossMinimizationCallback("loss"),
        }

    def get_engine(self):
        return DeviceEngine(self._device)

    def get_trial(self):
        return None

    def get_loggers(self):
        return {
            "console": dl.ConsoleLogger(),
            "csv": dl.CSVLogger(logdir="./logdir"),
        }


# execute whole experiment
class SupervisedRunner(IStageBasedRunner):
    def handle_batch(self, batch):
        x, y = batch
        logits = self.model(x)

        self.batch = {
            "features": x,
            "targets": y,
            "logits": logits,
        }


def run_train_with_experiment_device(device):
    # dataset = DummyDataset(10)
    # loader = DataLoader(dataset, batch_size=4)
    runner = SupervisedRunner()
    experiment = CustomExperiment(device)
    # experiment = Experiment(
    #     model=_model_fn,
    #     criterion=nn.MSELoss(),
    #     optimizer=_optimizer_fn,
    #     loaders={"train": loader, "valid": loader},
    #     main_metric="loss",
    #     callbacks=[
    #         CriterionCallback(),
    #         OptimizerCallback(),
    #         DeviceCheckCallback(device),
    #         LossMinimizationCallback("loss"),
    #     ],
    #     engine=DeviceEngine(device),
    # )
    runner.run(experiment)


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
