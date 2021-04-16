# flake8: noqa

from collections import OrderedDict
from io import StringIO
import os
import re
import shutil
import sys
from tempfile import TemporaryDirectory

import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

# local
import catalyst.dl as dl
from catalyst.engines import DataParallelEngine, DeviceEngine, DistributedDataParallelEngine
from catalyst.settings import IS_CUDA_AVAILABLE, NUM_CUDA_DEVICES

if NUM_CUDA_DEVICES > 1:
    os.environ["MKL_SERVICE_FORCE_INTEL"] = "1"


class DummyDataset:
    """Docs."""

    features_dim: int = 4
    out_dim: int = 2

    def __init__(self, num_records: int):
        """Docs."""
        self.num_records = num_records

    def __len__(self):
        """Docs."""
        return self.num_records

    def __getitem__(self, idx: int):
        """Docs."""
        x = torch.ones(self.features_dim, dtype=torch.float)
        y = torch.ones(self.out_dim, dtype=torch.float)
        return x, y


class CheckModelStateLoadAfterStages(dl.Callback):
    """Docs."""

    def __init__(self, stage, logdir, checkpoint):
        """Docs."""
        super().__init__(dl.CallbackOrder.Internal)
        self.stage = stage
        self.logdir = logdir
        self.checkpoint = checkpoint

    def on_stage_start(self, runner):
        if runner.stage_key != self.stage or not runner.engine.is_master_process:
            return
        # modify model state
        checkpoint_file = os.path.join(self.logdir, self.checkpoint)
        checkpoint = runner.engine.load_checkpoint(checkpoint_file)
        checkpoint["model_state_dict"] = OrderedDict(
            (k, torch.ones_like(v)) for k, v in checkpoint["model_state_dict"].items()
        )
        # print("-" * 100)
        # print(checkpoint)
        # print(runner.model.state_dict())
        # print("-" * 100)
        runner.engine.save_checkpoint(checkpoint, checkpoint_file)

    def on_batch_start(self, runner):
        if not (runner.stage_key == self.stage and runner.stage_batch_step == 0):
            return
        # check if model loaded right checkpoint
        model = runner.model
        if not isinstance(model, torch.nn.Module):  # dummy check for DP or DDP
            model = model.module
        state_dict = model.state_dict()
        # print("=" * 100)
        # print(model.state_dict())
        # print("=" * 100)
        for k, v in state_dict.items():
            assert torch.all(v.isclose(torch.ones_like(v))), (
                f"Stage: '{runner.stage_key}'\n"
                f"Epoch: {runner.stage_epoch_step} (global - {runner.global_epoch_step})\n"
                f"Batch: {runner.stage_batch_step} (global - {runner.global_batch_step})\n"
                f"Expected that value for '{k}' will be all ones!\n"
                f"Got:\n{v}\n"
            )


class CustomRunner(dl.IRunner):
    def __init__(self, logdir, engine):
        super().__init__()
        self._logdir = logdir
        self._engine = engine

    def get_engine(self):
        return self._engine

    def get_callbacks(self, stage: str):
        return {
            "criterion": dl.CriterionCallback(
                metric_key="loss", input_key="logits", target_key="targets"
            ),
            "optimizer": dl.OptimizerCallback(metric_key="loss"),
            "checkpoint": dl.CheckpointCallback(
                self._logdir,
                loader_key="valid",
                metric_key="loss",
                minimize=True,
                save_n_best=3,
                load_on_stage_start="best",
            ),
            "test_model_load": CheckModelStateLoadAfterStages("second", self._logdir, "best.pth"),
        }

    @property
    def stages(self) -> "Iterable[str]":
        return ["first", "second"]

    def get_stage_len(self, stage: str) -> int:
        return 10

    def get_loaders(self, stage: str):
        dataset = DummyDataset(64)
        loader = DataLoader(dataset, batch_size=4, num_workers=1)
        return {"train": loader, "valid": loader}

    def get_model(self, stage: str):
        return torch.nn.Linear(4, 2)

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


def test_device_load_on_stage_start():
    to_check_devices = ["cpu"]
    for device in to_check_devices:
        with TemporaryDirectory() as logdir:
            runner = CustomRunner(logdir, DeviceEngine(device))
            runner.run()


@pytest.mark.skipif(
    not IS_CUDA_AVAILABLE, reason="CUDA is not available",
)
def test_device_load_on_stage_start():
    to_check_devices = [f"cuda:{i}" for i in range(NUM_CUDA_DEVICES)]
    for device in to_check_devices:
        with TemporaryDirectory() as logdir:
            runner = CustomRunner(logdir, DeviceEngine(device))
            runner.run()


@pytest.mark.skipif(
    not (IS_CUDA_AVAILABLE and NUM_CUDA_DEVICES >= 2),
    reason="Number of CUDA devices is less than 2",
)
def test_dp_load_on_stage_start():
    with TemporaryDirectory() as logdir:
        runner = CustomRunner(logdir, DataParallelEngine())
        runner.run()


@pytest.mark.skipif(
    not (IS_CUDA_AVAILABLE and NUM_CUDA_DEVICES >= 2),
    reason="Number of CUDA devices is less than 2",
)
def test_ddp_load_on_stage_start():
    with TemporaryDirectory() as logdir:
        runner = CustomRunner(logdir, DistributedDataParallelEngine())
        runner.run()


def test_load_best_on_stage_end():
    old_stdout = sys.stdout
    sys.stdout = str_stdout = StringIO()

    # experiment_setup
    logdir = "./logs/checkpoint_callback"
    checkpoint = logdir  # + "/checkpoints"
    logfile = checkpoint + "/_metrics.json"

    # data
    num_samples, num_features = int(1e4), int(1e1)
    X = torch.rand(num_samples, num_features)
    y = torch.randint(0, 5, size=[num_samples])
    dataset = TensorDataset(X, y)
    loader = DataLoader(dataset, batch_size=32, num_workers=1)
    loaders = {"train": loader, "valid": loader}

    # model, criterion, optimizer, scheduler
    model = torch.nn.Linear(num_features, 5)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters())
    runner = dl.SupervisedRunner()

    n_epochs = 5
    # first stage
    runner.train(
        model=model,
        criterion=criterion,
        optimizer=optimizer,
        loaders=loaders,
        logdir=logdir,
        num_epochs=n_epochs,
        verbose=False,
        valid_loader="valid",
        valid_metric="loss",
        minimize_valid_metric=True,
        callbacks=[
            dl.CheckpointCallback(
                logdir=logdir,
                loader_key="valid",
                metric_key="loss",
                minimize=True,
                save_n_best=2,
                load_on_stage_end="best",
            ),
            dl.CheckRunCallback(num_epoch_steps=n_epochs),
        ],
    )

    sys.stdout = old_stdout
    exp_output = str_stdout.getvalue()

    assert len(re.findall(r"=> Loading", exp_output)) == 1
    assert len(re.findall(r"=> Loading .*best\.pth", exp_output)) == 1

    assert os.path.isfile(logfile)
    assert os.path.isfile(checkpoint + "/train.4.pth")
    assert os.path.isfile(checkpoint + "/train.4_full.pth")
    assert os.path.isfile(checkpoint + "/train.5.pth")
    assert os.path.isfile(checkpoint + "/train.5_full.pth")
    assert os.path.isfile(checkpoint + "/best.pth")
    assert os.path.isfile(checkpoint + "/best_full.pth")
    assert os.path.isfile(checkpoint + "/last.pth")
    assert os.path.isfile(checkpoint + "/last_full.pth")

    shutil.rmtree(logdir, ignore_errors=True)


# @pytest.mark.skip(reason="disabled")
# def test_multiple_stages_and_different_checkpoints_to_load():
#     old_stdout = sys.stdout
#     sys.stdout = str_stdout = StringIO()
#
#     # experiment_setup
#     logdir = "./logs/checkpoint_callback"
#     checkpoint = logdir  # + "/checkpoints"
#     logfile = checkpoint + "/_metrics.json"
#     num_epochs = 5
#
#     # data
#     num_samples, num_features = int(1e4), int(1e1)
#     X = torch.rand(num_samples, num_features)
#     y = torch.randint(0, 5, size=[num_samples])
#     dataset = TensorDataset(X, y)
#     loader = DataLoader(dataset, batch_size=32, num_workers=1)
#     loaders = {"train": loader, "valid": loader}
#
#     # model, criterion, optimizer, scheduler
#     model = torch.nn.Linear(num_features, 5)
#     criterion = torch.nn.CrossEntropyLoss()
#     optimizer = torch.optim.Adam(model.parameters())
#     runner = dl.SupervisedRunner()
#
#     # first stage
#     runner.train(
#         model=model,
#         criterion=criterion,
#         optimizer=optimizer,
#         loaders=loaders,
#         logdir=logdir,
#         num_epochs=num_epochs,
#         verbose=False,
#         valid_loader="valid",
#         valid_metric="loss",
#         minimize_valid_metric=True,
#         callbacks=[
#             dl.CheckpointCallback(
#                 logdir=logdir,
#                 loader_key="valid",
#                 metric_key="loss",
#                 minimize=True,
#                 save_n_best=2,
#                 load_on_stage_end={"model": "best", "criterion": "best", "optimizer": "last"},
#             ),
#             dl.CheckRunCallback(num_epoch_steps=num_epochs),
#         ],
#     )
#     # second stage
#     runner.train(
#         model=model,
#         criterion=criterion,
#         optimizer=optimizer,
#         loaders=loaders,
#         logdir=logdir,
#         num_epochs=num_epochs,
#         verbose=False,
#         valid_loader="valid",
#         valid_metric="loss",
#         minimize_valid_metric=True,
#         callbacks=[
#             dl.CheckpointCallback(
#                 logdir=logdir,
#                 loader_key="valid",
#                 metric_key="loss",
#                 minimize=True,
#                 save_n_best=3,
#                 load_on_stage_start={"model": "last", "criterion": "last", "optimizer": "best"},
#             ),
#             dl.CheckRunCallback(num_epoch_steps=num_epochs),
#         ],
#     )
#
#     sys.stdout = old_stdout
#     exp_output = str_stdout.getvalue()
#
#     assert len(re.findall(r"=> Loading", exp_output)) == 3
#     assert len(re.findall(r"=> Loading .*best_full\.pth", exp_output)) == 2
#     assert len(re.findall(r"=> Loading .*last_full\.pth", exp_output)) == 1
#
#     assert os.path.isfile(logfile)
#     assert os.path.isfile(checkpoint + "/train.3.pth")
#     assert os.path.isfile(checkpoint + "/train.3_full.pth")
#     assert os.path.isfile(checkpoint + "/train.4.pth")
#     assert os.path.isfile(checkpoint + "/train.4_full.pth")
#     assert os.path.isfile(checkpoint + "/train.5.pth")
#     assert os.path.isfile(checkpoint + "/train.5_full.pth")
#     assert os.path.isfile(checkpoint + "/best.pth")
#     assert os.path.isfile(checkpoint + "/best_full.pth")
#     assert os.path.isfile(checkpoint + "/last.pth")
#     assert os.path.isfile(checkpoint + "/last_full.pth")
#
#     shutil.rmtree(logdir, ignore_errors=True)
#
#
# @pytest.mark.skip(reason="disabled")
# def test_resume_with_missing_file():
#     old_stdout = sys.stdout
#     sys.stdout = str_stdout = StringIO()
#
#     # experiment_setup
#     logdir = "./logs/checkpoint_callback"
#     checkpoint = logdir + "/checkpoints"
#     logfile = checkpoint + "/_metrics.json"
#     num_epochs = 5
#
#     # data
#     num_samples, num_features = int(1e4), int(1e1)
#     X = torch.rand(num_samples, num_features)
#     y = torch.randint(0, 5, size=[num_samples])
#     dataset = TensorDataset(X, y)
#     loader = DataLoader(dataset, batch_size=32, num_workers=1)
#     loaders = {"train": loader, "valid": loader}
#
#     # model, criterion, optimizer, scheduler
#     model = torch.nn.Linear(num_features, 5)
#     criterion = torch.nn.CrossEntropyLoss()
#     optimizer = torch.optim.Adam(model.parameters())
#     runner = dl.SupervisedRunner()
#
#     with pytest.raises(FileNotFoundError):
#         runner.train(
#             model=model,
#             criterion=criterion,
#             optimizer=optimizer,
#             loaders=loaders,
#             logdir=logdir,
#             num_epochs=num_epochs,
#             verbose=False,
#             valid_loader="valid",
#             valid_metric="loss",
#             minimize_valid_metric=True,
#             callbacks=[
#                 dl.CheckpointCallback(
#                     logdir=logdir,
#                     loader_key="valid",
#                     metric_key="loss",
#                     minimize=True,
#                     save_n_best=2,
#                     load_on_stage_end={"model": "best", "criterion": "best", "optimizer": "last"},
#                     resume="not_existing_file.pth",
#                 ),
#                 dl.CheckRunCallback(num_epoch_steps=num_epochs),
#             ],
#         )
#
#     sys.stdout = old_stdout
#     exp_output = str_stdout.getvalue()
#
#     shutil.rmtree(logdir, ignore_errors=True)
#
#
# @pytest.mark.skip(reason="disabled")
# def test_load_on_stage_start_with_empty_dict():
#     old_stdout = sys.stdout
#     sys.stdout = str_stdout = StringIO()
#
#     # experiment_setup
#     logdir = "./logs/checkpoint_callback"
#     checkpoint = logdir  # + "/checkpoints"
#     logfile = checkpoint + "/_metrics.json"
#     num_epochs = 5
#
#     # data
#     num_samples, num_features = int(1e4), int(1e1)
#     X = torch.rand(num_samples, num_features)
#     y = torch.randint(0, 5, size=[num_samples])
#     dataset = TensorDataset(X, y)
#     loader = DataLoader(dataset, batch_size=32, num_workers=1)
#     loaders = {"train": loader, "valid": loader}
#
#     # model, criterion, optimizer, scheduler
#     model = torch.nn.Linear(num_features, 5)
#     criterion = torch.nn.CrossEntropyLoss()
#     optimizer = torch.optim.Adam(model.parameters())
#     runner = dl.SupervisedRunner()
#
#     # first stage
#     runner.train(
#         model=model,
#         criterion=criterion,
#         optimizer=optimizer,
#         loaders=loaders,
#         logdir=logdir,
#         num_epochs=num_epochs,
#         verbose=False,
#         valid_loader="valid",
#         valid_metric="loss",
#         minimize_valid_metric=True,
#         callbacks=[
#             dl.CheckpointCallback(
#                 logdir=logdir, loader_key="valid", metric_key="loss", minimize=True, save_n_best=2
#             ),
#             dl.CheckRunCallback(num_epoch_steps=num_epochs),
#         ],
#     )
#     # second stage
#     runner.train(
#         model=model,
#         criterion=criterion,
#         optimizer=optimizer,
#         loaders=loaders,
#         logdir=logdir,
#         num_epochs=num_epochs,
#         verbose=False,
#         valid_loader="valid",
#         valid_metric="loss",
#         minimize_valid_metric=True,
#         callbacks=[
#             dl.CheckpointCallback(
#                 logdir=logdir,
#                 loader_key="valid",
#                 metric_key="loss",
#                 minimize=True,
#                 save_n_best=3,
#                 load_on_stage_start={},
#             ),
#             dl.CheckRunCallback(num_epoch_steps=num_epochs),
#         ],
#     )
#
#     sys.stdout = old_stdout
#     exp_output = str_stdout.getvalue()
#
#     assert len(re.findall(r"=> Loading", exp_output)) == 0
#
#     assert os.path.isfile(logfile)
#     assert os.path.isfile(checkpoint + "/train.3.pth")
#     assert os.path.isfile(checkpoint + "/train.3_full.pth")
#     assert os.path.isfile(checkpoint + "/train.4.pth")
#     assert os.path.isfile(checkpoint + "/train.4_full.pth")
#     assert os.path.isfile(checkpoint + "/train.5.pth")
#     assert os.path.isfile(checkpoint + "/train.5_full.pth")
#     assert os.path.isfile(checkpoint + "/best.pth")
#     assert os.path.isfile(checkpoint + "/best_full.pth")
#     assert os.path.isfile(checkpoint + "/last.pth")
#     assert os.path.isfile(checkpoint + "/last_full.pth")
#
#     shutil.rmtree(logdir, ignore_errors=True)
