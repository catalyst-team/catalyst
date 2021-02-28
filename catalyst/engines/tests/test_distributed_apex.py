# flake8: noqa

from typing import Any, Dict, List
import logging
import os
from tempfile import TemporaryDirectory

from pytest import mark
import torch
from torch.utils.data import DataLoader

from catalyst import dl
from catalyst.settings import IS_APEX_AVAILABLE, IS_CUDA_AVAILABLE, NUM_CUDA_DEVICES

if IS_APEX_AVAILABLE:
    from catalyst.engines.apex import DistributedDataParallelApexEngine

from .misc import (
    DummyDataset,
    DummyModel,
    LossMinimizationCallback,
    OPTTensorTypeChecker,
    WorldSizeCheckCallback,
)

logger = logging.getLogger(__name__)


if NUM_CUDA_DEVICES > 1:
    os.environ["MKL_SERVICE_FORCE_INTEL"] = "1"


OPT_LEVELS = ("O0", "O1", "O2", "O3")


class CustomRunner(dl.IRunner):
    def __init__(self, logdir, opt_level, port="12345"):
        super().__init__()
        self._logdir = logdir
        self._opt_level = opt_level
        self._port = port

    def get_engine(self) -> dl.IEngine:
        return DistributedDataParallelApexEngine(port=self._port, opt_level=self._opt_level)

    def get_callbacks(self, stage: str) -> Dict[str, dl.Callback]:
        return {
            "criterion": dl.CriterionCallback(metric_key="loss", input_key="logits", target_key="targets"),
            "optimizer": dl.OptimizerCallback(metric_key="loss"),
            # "scheduler": dl.SchedulerCallback(loader_key="valid", metric_key="loss"),
            # "checkpoint": dl.CheckpointCallback(
            #     self._logdir, loader_key="valid", metric_key="loss", minimize=True, save_n_best=3
            # ),
            # "check": DeviceCheckCallback(),
            "check2": LossMinimizationCallback("loss", logger=logger),
            "check_world_size": WorldSizeCheckCallback(NUM_CUDA_DEVICES, logger=logger),
            "logits_type_checker": OPTTensorTypeChecker("logits", self._opt_level),
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

    def get_scheduler(self, optimizer, stage: str):
        return None

    def get_trial(self):
        return None

    def get_loggers(self):
        return {"console": dl.ConsoleLogger(), "csv": dl.CSVLogger(logdir=self._logdir)}

    def handle_batch(self, batch):
        x, y = batch
        logits = self.model(x)

        self.batch = {"features": x, "targets": y, "logits": logits}


@mark.skipif(
    not IS_CUDA_AVAILABLE and NUM_CUDA_DEVICES < 2, reason="Number of CUDA devices is less than 2",
)
def test_train_distributed_parallel_apex():
    for idx, opt_level in enumerate(OPT_LEVELS):
        with TemporaryDirectory() as logdir:
            runner = CustomRunner(logdir, opt_level, str(12345 + idx))
            runner.run()


@mark.skip("Config experiment is in development phase!")
@mark.skipif(
    not IS_CUDA_AVAILABLE and NUM_CUDA_DEVICES < 2, reason="Number of CUDA devices is less than 2",
)
def test_config_train_distributed_parallel_apex():
    pass
    # logdir = "./test_config_ddp_engine"
    # runner = SupervisedRunner()
    # exp = ConfigExperiment(
    #     config={
    #         "model_params": {"_target_": "DummyModel", "in_features": 4, "out_features": 1},
    #         "engine": "ddp",
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
    #                     # "test_device": {
    #                     #     "_target_": "DeviceCheckCallback",
    #                     #     "assert_device": str(device),
    #                     # },
    #                     "test_loss_minimization": {"_target_": "LossMinimizationCallback"},
    #                 },
    #             },
    #         },
    #     }
    # )
    # exp.get_loaders = _get_loaders
    # # CORE
    # runner.run_experiment(exp)
    # shutil.rmtree(logdir, ignore_errors=True)
