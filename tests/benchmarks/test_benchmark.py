# flake8: noqa
from typing import Any, Mapping
from collections import OrderedDict
import enum
import gc
import os
import time

import numpy as np
import pytest
from tqdm import tqdm

import torch
from torch import nn
from torch.utils.data.dataloader import DataLoader

from catalyst import dl, utils
from catalyst.contrib.datasets import MNIST
from catalyst.typing import TorchCriterion, TorchModel, TorchOptimizer
from tests import DATA_ROOT

IS_BENCHMARK_REQUIRED = os.environ.get("BENCHMARK_REQUIRED", "0") == "1"


class RunMode(str, enum.Enum):
    """RunModes."""

    catalyst = "catalyst"
    pytorch = "pytorch"


class TestMnistRunner(dl.Runner):
    def get_loaders(self) -> "OrderedDict[str, DataLoader]":
        return {
            "train": DataLoader(
                MNIST(DATA_ROOT, train=True, download=True),
                batch_size=128,
                num_workers=1,
            )
        }

    def get_model(self) -> TorchModel:
        return nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=28 * 28, out_features=128),
            nn.BatchNorm1d(128),
            nn.Dropout(0.5),
            nn.ReLU(),
            nn.Linear(in_features=128, out_features=10),
        )

    def get_criterion(self) -> TorchCriterion:
        return nn.CrossEntropyLoss()

    def get_optimizer(self, model: TorchModel) -> TorchOptimizer:
        return torch.optim.Adam(model.parameters(), lr=0.02)

    def handle_batch(self, batch: Mapping[str, Any]) -> None:
        raise NotImplementedError()


def _get_used_memory():
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        used_memory = torch.cuda.max_memory_allocated()
    else:
        used_memory = np.nan
    return used_memory


def run_pytorch(
    irunner: dl.IRunner, idx: int, device: str = "cuda", num_epochs: int = 10
):
    device = torch.device(device)
    utils.set_global_seed(idx)

    loader = irunner.get_loaders()["train"]
    model = irunner.get_model().to(device)
    criterion = irunner.get_criterion()
    optimizer = irunner.get_optimizer(model)

    epoch_scores = []
    epoch_losses = []
    for i in range(num_epochs):
        epoch_score = 0
        epoch_loss = 0

        for features, targets in loader:
            features = features.to(device)
            targets = targets.to(device)
            logits = model(features)
            loss = criterion(logits, targets)

            epoch_loss += loss.item()
            pred = logits.argmax(dim=1, keepdim=True)
            epoch_score += pred.eq(targets.view_as(pred)).sum().item()

            self.engine.backward(loss)
            optimizer.step()
            optimizer.zero_grad()

        epoch_score /= len(loader.dataset)
        epoch_loss /= len(loader)

        print(f"Epoch {i} \t Score: {epoch_score} \t Loss: {epoch_loss}")

        epoch_scores.append(epoch_score)
        epoch_losses.append(epoch_loss)

    return epoch_scores[-1], epoch_losses[-1], _get_used_memory()


def run_catalyst(
    irunner: dl.IRunner, idx: int, device: str = "cuda", num_epochs: int = 10
):
    utils.set_global_seed(idx)
    loader = irunner.get_loaders()["train"]
    model = irunner.get_model().to(device)
    criterion = irunner.get_criterion()
    optimizer = irunner.get_optimizer(model)

    runner = dl.SupervisedRunner()
    runner.train(
        engine=dl.GPUEngine() if device == "cuda" else dl.CPUEngine(),
        model=model,
        criterion=criterion,
        optimizer=optimizer,
        loaders={"train": loader},
        num_epochs=num_epochs,
        verbose=False,
        callbacks=[
            dl.AccuracyCallback(
                input_key=runner._output_key,
                target_key=runner._target_key,
                topk=(1,),
            )
        ],
    )

    return (
        runner.epoch_metrics["train"]["accuracy01"],
        runner.epoch_metrics["train"]["loss"],
        _get_used_memory(),
    )


def score_runs(
    irunner: dl.IRunner,
    mode: RunMode,
    device: str,
    num_runs: int = 10,
    num_epochs: int = 10,
):
    hist_scores = []
    hist_losses = []
    hist_time = []
    hist_memory = []

    torch.backends.cudnn.deterministic = True
    for i in tqdm(range(num_runs), desc=f"{mode} with {irunner.__class__.__name__}"):
        gc.collect()
        if device == "cuda":
            torch.cuda.empty_cache()
            torch.cuda.reset_max_memory_cached()
            torch.cuda.reset_max_memory_allocated()
            torch.cuda.reset_accumulated_memory_stats()
            torch.cuda.reset_peak_memory_stats()
        time.sleep(1)

        time_start = time.perf_counter()

        _run_fn = run_catalyst if mode == RunMode.catalyst else run_pytorch
        final_score, final_loss, used_memory = _run_fn(
            irunner, idx=i, device=device, num_epochs=num_epochs
        )

        time_end = time.perf_counter()

        hist_scores.append(final_score)
        hist_losses.append(final_loss)
        hist_time.append(time_end - time_start)
        hist_memory.append(used_memory)

    return {
        "scores": hist_scores,
        "losses": hist_losses,
        "time": hist_time,
        "memory": hist_memory,
    }


def assert_relative_equal(
    catalyst_values, torch_values, max_diff: float, norm: float = 1
):
    diffs = np.asarray(catalyst_values) - np.mean(torch_values)
    diffs = diffs / norm
    diffs = diffs / np.mean(torch_values)
    assert (
        np.mean(diffs) < max_diff
    ), f"Catalyst diff {diffs} worse than PyTorch (threshold {max_diff})"


def assert_absolute_equal(
    catalyst_values, torch_values, max_diff: float, norm: float = 1
):
    diffs = np.asarray(catalyst_values) - np.mean(torch_values)
    diffs = diffs / norm
    assert (
        np.mean(diffs) < max_diff
    ), f"Catalyst {diffs} worse than PyTorch (threshold {max_diff})"


BENCHMARKS = [(TestMnistRunner, 4, "cpu", 3, 2, 0.15, 0.001)]
if torch.cuda.is_available():
    BENCHMARKS.append((TestMnistRunner, 4, "cuda", 3, 2, 0.15, 0.001))


@pytest.mark.parametrize(
    "irunner,num_epochs,device,num_runs,precision,max_diff_time,max_diff_memory",
    BENCHMARKS,
)
@pytest.mark.skipif(~IS_BENCHMARK_REQUIRED, reason="Benchmark is not required.")
def test_benchmark(
    tmpdir,
    irunner: dl.IRunner,
    device: str,
    num_epochs: int,
    num_runs: int,
    precision: int,
    max_diff_time: float,
    max_diff_memory: float,
):

    irunner = irunner()
    # prepare data
    _ = irunner.get_loaders()

    # score runs
    pytorch = score_runs(
        irunner,
        mode=RunMode.pytorch,
        device=device,
        num_epochs=num_epochs,
        num_runs=num_runs,
    )
    catalyst = score_runs(
        irunner,
        mode=RunMode.catalyst,
        device=device,
        num_epochs=num_epochs,
        num_runs=num_runs,
    )

    # check performance
    print(
        "Scores are for... \n "
        f"PyTorch: {pytorch['scores']} \n Catalyst: {catalyst['scores']}"
    )
    for catalyst_, pytorch_ in zip(catalyst["scores"], pytorch["scores"]):
        np.testing.assert_almost_equal(catalyst_, pytorch_, precision)

    # check loss
    print(
        "Losses are for... \n "
        f"PyTorch: {pytorch['losses']} \n Catalyst: {catalyst['losses']}"
    )
    for catalyst_, pytorch_ in zip(catalyst["losses"], pytorch["losses"]):
        np.testing.assert_almost_equal(catalyst_, pytorch_, precision)

    # check time
    print(
        f"Times are for... \n PyTorch: {pytorch['time']} \n Catalyst: {catalyst['time']}"
    )
    assert_absolute_equal(
        catalyst["time"],
        pytorch["time"],
        norm=num_epochs,
        max_diff=max_diff_time,
    )

    # check memory
    if torch.cuda.is_available():
        print(
            "Memory usages are for... \n "
            f"PyTorch: {pytorch['memory']} \n Catalyst: {catalyst['memory']}"
        )
        assert_relative_equal(
            catalyst["memory"], pytorch["memory"], max_diff=max_diff_memory
        )
