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
from catalyst.engines.apex import APEXEngine
from catalyst.settings import IS_CUDA_AVAILABLE, NUM_CUDA_DEVICES

from .test_device import DummyDataset, DummyModel, LossMinimizationCallback  # SupervisedRunner,

logger = logging.getLogger(__name__)


OPT_LEVELS = ("O0", "O1", "O2", "O3")
OPT_TYPE_MAP = {
    "O0": torch.float32,  # no-op training
    "O1": torch.float16,  # mixed precision (FP16) training
    "O2": torch.float32,  # almost FP16 training
    "O3": torch.float32,  # another implementation of FP16 training
}


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
    def __init__(self, key, opt_level, use_batch_metrics=False):
        super().__init__(dl.CallbackOrder.Metric)
        self.key = key
        self.use_batch_metrics = use_batch_metrics
        self.opt_level = opt_level
        self.expected_type = OPT_TYPE_MAP[opt_level]

    def on_batch_end(self, runner):
        check_tensor = (
            runner.batch_metrics[self.key] if self.use_batch_metrics else runner.batch[self.key]
        )
        assert check_tensor.dtype == self.expected_type, (
            f"Wrong types for {self.opt_level} - actual is "
            f"'{check_tensor.dtype}' but expected is '{self.expected_type}'!"
        )


class CustomExperiment(dl.IExperiment):
    _logdir = "./logdir"

    def __init__(self, device, opt_level):
        self._device = device
        self._opt_level = opt_level

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
            # TODO: fix issue with pickling wrapped model's forward function
            # "checkpoint": dl.CheckpointCallback(
            #     self._logdir, loader_key="valid", metric_key="loss", minimize=True, save_n_best=3
            # ),
            # "check": DeviceCheckCallback(),
            "check2": LossMinimizationCallback("loss"),
            "logits_type_checker": TensorTypeChecker("logits", self._opt_level),
            # "loss_type_checker": TensorTypeChecker("loss", True),
        }

    def get_engine(self):
        return APEXEngine(self._device, self._opt_level)

    def get_trial(self):
        return None

    def get_loggers(self):
        return {
            "console": dl.ConsoleLogger(),
            "csv": dl.CSVLogger(logdir=self._logdir),
        }


def run_train_with_experiment_apex_device(device, opt_level):
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
        experiment = CustomExperiment(device, opt_level)
        experiment._logdir = logdir
        runner.run(experiment)


@mark.skipif(not IS_CUDA_AVAILABLE, reason="CUDA device is not available")
def test_apex_with_cuda():
    for level in OPT_LEVELS:
        run_train_with_experiment_apex_device("cuda:0", level)


@mark.skip("Config experiment is in development phase!")
@mark.skipif(not IS_CUDA_AVAILABLE, reason="CUDA device is not available")
def test_config_apex_with_cuda():
    for level in OPT_LEVELS:
        run_train_with_experiment_apex_device("cuda:0", level)


@mark.skipif(
    not IS_CUDA_AVAILABLE and NUM_CUDA_DEVICES < 2, reason="Number of CUDA devices is less than 2",
)
def test_apex_with_other_cuda_device():
    for level in OPT_LEVELS:
        run_train_with_experiment_apex_device("cuda:1", level)


@mark.skip("Config experiment is in development phase!")
@mark.skipif(
    not IS_CUDA_AVAILABLE and NUM_CUDA_DEVICES < 2, reason="Number of CUDA devices is less than 2",
)
def test_config_apex_with_other_cuda_device():
    for level in OPT_LEVELS:
        run_train_with_experiment_apex_device("cuda:1", level)
