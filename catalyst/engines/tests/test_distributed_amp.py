# flake8: noqa

from typing import Any, Dict, List
import os
from tempfile import TemporaryDirectory

from pytest import mark

import torch
from torch.utils.data import DataLoader

from catalyst import dl
from catalyst.engines.amp import DistributedDataParallelAMPEngine
from catalyst.settings import IS_CUDA_AVAILABLE, NUM_CUDA_DEVICES
from .utils import DummyDataset, DummyModel, LossMinimizationCallback, WorldSizeCheckCallback

if NUM_CUDA_DEVICES > 1:
    os.environ["MKL_SERVICE_FORCE_INTEL"] = "1"


class CustomRunner(dl.IRunner):
    def __init__(self, logdir):
        super().__init__()
        self._logdir = logdir

    def get_engine(self) -> dl.IEngine:
        return DistributedDataParallelAMPEngine()

    def get_callbacks(self, stage: str) -> Dict[str, dl.Callback]:
        return {
            "criterion": dl.CriterionCallback(metric_key="loss", input_key="logits", target_key="targets"),
            "optimizer": dl.OptimizerCallback(metric_key="loss"),
            # "scheduler": dl.SchedulerCallback(loader_key="valid", metric_key="loss"),
            # "checkpoint": dl.CheckpointCallback(
            #     self._logdir, loader_key="valid", metric_key="loss", minimize=True, save_n_best=3
            # ),
            # "check": DeviceCheckCallback(),
            "check2": LossMinimizationCallback("loss"),
            "check_world_size": WorldSizeCheckCallback(NUM_CUDA_DEVICES),
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


@mark.skipif(
    not IS_CUDA_AVAILABLE and NUM_CUDA_DEVICES < 2, reason="Number of CUDA devices is less than 2",
)
def test_train_with_experiment_distributed_parallel_amp_device():
    # logdir = "./test_ddp_engine"
    # dataset = DummyDataset(10)
    # # sampler = DistributedSampler(dataset, world_size, rank)
    # loader = DataLoader(dataset, batch_size=4)  # , sampler=sampler)
    # runner = SupervisedRunner()
    # engine = DistributedDataParallelEngine()
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
    #     logdir=logdir,
    #     engine=engine,
    # )
    with TemporaryDirectory() as logdir:
        runner = CustomRunner(logdir)
        runner.run()


@mark.skip("Config experiment is in development phase!")
@mark.skipif(
    not IS_CUDA_AVAILABLE and NUM_CUDA_DEVICES < 2, reason="Number of CUDA devices is less than 2",
)
def test_train_with_config_experiment_distributed_parallel_amp_device():
    pass
    # logdir = "./test_config_ddp_engine"
    # runner = SupervisedRunner()
    # exp = ConfigExperiment(
    #     config={
    #         "model_params": {"_target_": "DummyModel", "in_features": 4, "out_features": 1,},
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
