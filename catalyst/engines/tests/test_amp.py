# flake8: noqa

from typing import Any, Dict, List
import logging
from tempfile import TemporaryDirectory

from pytest import mark
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from catalyst import dl
from catalyst.engines.amp import AMPEngine
from catalyst.settings import IS_CUDA_AVAILABLE, NUM_CUDA_DEVICES

from .test_device import DummyDataset, DummyModel, LossMinimizationCallback  # SupervisedRunner,

logger = logging.getLogger(__name__)


class SupervisedRunner(dl.IStageBasedRunner):
    def handle_batch(self, batch):
        x, y = batch

        logits = self.model(x)

        logger.warning(f"x dtype: {x.dtype}")
        logger.warning(f"y dtype: {y.dtype}")
        logger.warning(f"logits dtype: {logits.dtype}")

        self.batch = {
            "features": x,
            "targets": y,
            "logits": logits,
        }


class TensorTypeChecker(dl.Callback):
    def __init__(self, key, use_batch_metrics=False):
        super().__init__(dl.CallbackOrder.Metric)
        self.key = key
        self.use_batch_metrics = use_batch_metrics

    def on_batch_end(self, runner):
        if self.use_batch_metrics:
            assert runner.batch_metrics[self.key].dtype == torch.float16
        else:
            assert runner.batch[self.key].dtype == torch.float16


class CustomExperiment(dl.IExperiment):
    _logdir = "./logdir"

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
        dataset = DummyDataset(10)
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
                self._logdir, loader_key="valid", metric_key="loss", minimize=True, save_n_best=3
            ),
            # "check": DeviceCheckCallback(),
            "check2": LossMinimizationCallback("loss"),
            "logits_type_checker": TensorTypeChecker("logits"),
            # "loss_type_checker": TensorTypeChecker("loss", True),
        }

    def get_engine(self):
        return AMPEngine(self._device)

    def get_trial(self):
        return None

    def get_loggers(self):
        return {
            "console": dl.ConsoleLogger(),
            "csv": dl.CSVLogger(logdir=self._logdir),
        }


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
        runner = SupervisedRunner()
        experiment = CustomExperiment(device)
        experiment._logdir = logdir
        runner.run(experiment)


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
