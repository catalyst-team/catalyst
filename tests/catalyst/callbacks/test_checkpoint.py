# flake8: noqa
# TODO: add test for `save_n_best=0``

from collections import OrderedDict
import os
import re

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
        if not (runner.stage_key == self.stage and runner.stage_batch_step == 1):
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
        callbacks = {
            "criterion": dl.CriterionCallback(
                metric_key="loss", input_key="logits", target_key="targets"
            ),
            "optimizer": dl.OptimizerCallback(metric_key="loss"),
            "test_model_load": CheckModelStateLoadAfterStages("second", self._logdir, "best.pth"),
        }
        if stage == "first":
            callbacks["checkpoint"] = dl.CheckpointCallback(
                self._logdir,
                loader_key="valid",
                metric_key="loss",
                minimize=True,
                save_n_best=3,
            )
        elif stage == "second":
            callbacks["checkpoint"] = dl.CheckpointCallback(
                self._logdir,
                loader_key="valid",
                metric_key="loss",
                minimize=True,
                save_n_best=3,
                load_on_stage_start="best",
            )
        return callbacks

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
        return {}

    def handle_batch(self, batch):
        x, y = batch
        logits = self.model(x)

        self.batch = {"features": x, "targets": y, "logits": logits}


@pytest.mark.parametrize(
    "device",
    [
        "cpu",
        pytest.param(
            "cuda", marks=pytest.mark.skipif(not IS_CUDA_AVAILABLE, reason="CUDA is not available")
        ),
    ],
)
def test_device_load_on_stage_end(device, tmpdir):
    logdir = tmpdir
    runner = CustomRunner(logdir, DeviceEngine(device))
    runner.run()


@pytest.mark.skipif(
    not (IS_CUDA_AVAILABLE and NUM_CUDA_DEVICES >= 2),
    reason="Number of CUDA devices is less than 2",
)
def test_dp_load_on_stage_end(tmpdir):
    logdir = tmpdir
    runner = CustomRunner(logdir, DataParallelEngine())
    runner.run()


@pytest.mark.skipif(
    not (IS_CUDA_AVAILABLE and NUM_CUDA_DEVICES >= 2),
    reason="Number of CUDA devices is less than 2",
)
def test_ddp_load_on_stage_start(tmpdir):
    logdir = tmpdir
    runner = CustomRunner(logdir, DistributedDataParallelEngine())
    runner.run()


def train_runner(logdir, n_epochs, callbacks):
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
    runner.loggers = {}

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
        callbacks=callbacks,
    )
    return runner


def test_files_existence(tmpdir):
    logfile = tmpdir + "/_metrics.json"
    n_epochs = 5
    callbacks = [
        dl.CheckpointCallback(
            logdir=tmpdir,
            loader_key="valid",
            metric_key="loss",
            minimize=True,
            save_n_best=2,
        ),
        dl.CheckRunCallback(num_epoch_steps=n_epochs),
    ]
    train_runner(tmpdir, n_epochs, callbacks)

    assert os.path.isfile(logfile)
    assert os.path.isfile(tmpdir + "/train.4.pth")
    assert os.path.isfile(tmpdir + "/train.4_full.pth")
    assert os.path.isfile(tmpdir + "/train.5.pth")
    assert os.path.isfile(tmpdir + "/train.5_full.pth")
    assert os.path.isfile(tmpdir + "/best.pth")
    assert os.path.isfile(tmpdir + "/best_full.pth")
    assert os.path.isfile(tmpdir + "/last.pth")
    assert os.path.isfile(tmpdir + "/last_full.pth")


@pytest.mark.parametrize(("to_load", "exp_loaded"), [("best", "model"), ("best_full", "full")])
def test_load_str_on_stage_end(to_load, exp_loaded, capsys, tmpdir):
    # experiment_setup
    n_epochs = 5
    callbacks = [
        dl.CheckpointCallback(
            logdir=tmpdir,
            loader_key="valid",
            metric_key="loss",
            minimize=True,
            save_n_best=2,
            load_on_stage_end=to_load,
        ),
        dl.CheckRunCallback(num_epoch_steps=n_epochs),
    ]

    train_runner(tmpdir, n_epochs, callbacks)
    exp_output = capsys.readouterr().out

    assert len(re.findall(r"=> Loading", exp_output)) == 1
    assert len(re.findall(r"=> Loading .*{}\.pth".format(to_load), exp_output)) == 1
    assert len(re.findall(r"{} checkpoint".format(exp_loaded), exp_output)) == 1


@pytest.mark.parametrize(
    ("to_load", "exp_loaded"),
    [
        ({"model": "best", "criterion": "best", "optimizer": "last"}, "model, criterion"),
        (
            {"model": "best", "criterion": "best", "optimizer": "best"},
            "model, criterion, optimizer",
        ),
    ],
)
def test_load_dict_on_stage_end(to_load, exp_loaded, capsys, tmpdir):
    # experiment_setup
    n_epochs = 5
    callbacks = [
        dl.CheckpointCallback(
            logdir=tmpdir,
            loader_key="valid",
            metric_key="loss",
            minimize=True,
            save_n_best=2,
            load_on_stage_end=to_load,
        ),
        dl.CheckRunCallback(num_epoch_steps=n_epochs),
    ]

    train_runner(tmpdir, n_epochs, callbacks)
    exp_output = capsys.readouterr().out

    assert len(re.findall(r"=> Loading", exp_output)) == 1
    assert len(re.findall(r"loaded: {}".format(exp_loaded), exp_output)) == 1


@pytest.mark.parametrize("to_load", [{}, None])
def test_load_empty(to_load, capsys, tmpdir):
    # experiment_setup
    n_epochs = 5
    callbacks = [
        dl.CheckpointCallback(
            logdir=tmpdir,
            loader_key="valid",
            metric_key="loss",
            minimize=True,
            save_n_best=2,
            load_on_stage_start=to_load,
            load_on_stage_end=to_load,
            resume=to_load,
        ),
        dl.CheckRunCallback(num_epoch_steps=n_epochs),
    ]

    train_runner(tmpdir, n_epochs, callbacks)
    exp_output = capsys.readouterr().out

    assert len(re.findall(r"=> Loading", exp_output)) == 0


@pytest.mark.parametrize(
    "to_load", ["best", {"model": "not_existing_file.pth", "criterion": "not_existing_file.pth"}]
)
def test_resume_with_missing_file(to_load, tmpdir):
    n_epochs = 5
    callbacks = [
        dl.CheckpointCallback(
            logdir=tmpdir,
            loader_key="valid",
            metric_key="loss",
            minimize=True,
            save_n_best=2,
            load_on_stage_start=to_load,
            load_on_stage_end=to_load,
            resume="best",
        ),
        dl.CheckRunCallback(num_epoch_steps=n_epochs),
    ]

    with pytest.raises(FileNotFoundError):
        train_runner(tmpdir, n_epochs, callbacks)
