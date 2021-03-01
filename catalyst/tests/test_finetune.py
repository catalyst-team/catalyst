# flake8: noqa

from typing import Any, Dict, List
import logging
from tempfile import TemporaryDirectory

from pytest import mark
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset

from catalyst import dl, utils
from catalyst.engines.device import DeviceEngine
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
class DummyModelFinetune(nn.Module):
    def __init__(self, in_features, hidden_features, out_features):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.layer1 = nn.Linear(in_features, hidden_features)
        self.layer2 = nn.Linear(hidden_features, out_features)

    def forward(self, batch):
        x = self.layer1(batch)
        x = F.relu(x)
        x = self.layer2(x)
        return x


class CheckRequiresGrad(dl.Callback):
    def __init__(self, layer_name, stage, requires_grad=True):
        super().__init__(dl.CallbackOrder.internal)
        self.name = layer_name
        self.stage = stage
        self.requires_grad = requires_grad

    def check_fn(self, model, stage):
        if stage != self.stage:
            return
        for layer_name, layer in model.named_children():
            if self.name != layer_name:
                continue
            for param in layer.parameters():
                assert (
                    self.requires_grad == param.requires_grad
                ), f"Stage '{stage}', layer '{self.name}': expected - {self.requires_grad}, actual - {param.requires_grad}"

    def on_stage_start(self, runner: dl.IRunner):
        self.check_fn(runner.model, runner.stage_key)

    def on_batch_start(self, runner: dl.IRunner):
        self.check_fn(runner.model, runner.stage_key)

    def on_batch_end(self, runner: dl.IRunner):
        self.check_fn(runner.model, runner.stage_key)

    def on_stage_enf(self, runner: dl.IRunner):
        self.check_fn(runner.model, runner.stage_key)


class CustomRunner(dl.IRunner):
    def __init__(self, logdir, device):
        super().__init__()
        self._logdir = logdir
        self._device = device

    def get_engine(self):
        return DeviceEngine(self._device)

    def get_loggers(self):
        return {"console": dl.ConsoleLogger(), "csv": dl.CSVLogger(logdir=self._logdir)}

    @property
    def stages(self) -> List[str]:
        return ["train_freezed", "train_unfreezed"]

    def get_stage_len(self, stage: str) -> int:
        return 3

    def get_loaders(self, stage: str, epoch: int = None) -> Dict[str, Any]:
        dataset = DummyDataset(6)
        loader = DataLoader(dataset, batch_size=4)
        return {"train": loader, "valid": loader}

    def get_model(self, stage: str):
        if self.model is not None:
            model = self.model
        else:
            model = DummyModelFinetune(4, 3, 2)
        if stage == "train_freezed":
            # freeze layer
            utils.set_requires_grad(model.layer1, False)
        else:
            utils.set_requires_grad(model, True)
        return model

    def get_criterion(self, stage: str):
        return nn.MSELoss()

    def get_optimizer(self, stage: str, model):
        return optim.Adam(model.parameters(), lr=1e-3)

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
                self._logdir, loader_key="valid", metric_key="loss", minimize=True, save_n_best=3
            ),
            "check_freezed": CheckRequiresGrad("layer1", "train_freezed", False),
            "check_unfreezed": CheckRequiresGrad("layer1", "train_unfreezed", True),
        }

    def handle_batch(self, batch):
        x, y = batch
        logits = self.model(x)

        self.batch = {
            "features": x,
            "targets": y,
            "logits": logits,
        }


def train_experiment(device):
    with TemporaryDirectory() as logdir:
        runner = CustomRunner(logdir, device)
        runner.run()


def train_config_experiment(device):
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


def test_finetune_on_cpu():
    train_experiment("cpu")


@mark.skip("Config experiment is in development phase!")
def test_finetune_on_config_cpu():
    train_config_experiment("cpu")


@mark.skipif(not IS_CUDA_AVAILABLE, reason="CUDA device is not available")
def test_finetune_on_cuda():
    train_experiment("cuda:0")


@mark.skip("Config experiment is in development phase!")
@mark.skipif(not IS_CUDA_AVAILABLE, reason="CUDA device is not available")
def test_finetune_on_config_cuda():
    train_config_experiment("cuda:0")


@mark.skipif(
    not IS_CUDA_AVAILABLE and NUM_CUDA_DEVICES < 2, reason="Number of CUDA devices is less than 2",
)
def test_finetune_on_cuda_device():
    train_experiment("cuda:1")


@mark.skip("Config experiment is in development phase!")
@mark.skipif(
    not IS_CUDA_AVAILABLE and NUM_CUDA_DEVICES < 2, reason="Number of CUDA devices is less than 2",
)
def test_finetune_on_config_cuda_device():
    train_config_experiment("cuda:1")
