# flake8: noqa
from typing import Dict, Iterable, Union
from collections import OrderedDict
from tempfile import TemporaryDirectory

import numpy as np
import pytest

import torch
from torch import nn
from torch.optim import Adam
from torch.utils.data import DataLoader, TensorDataset

from catalyst import dl
from catalyst.contrib.data import AllTripletsSampler, HardTripletsSampler
from catalyst.contrib.datasets import MnistMLDataset, MnistQGDataset
from catalyst.contrib.losses import TripletMarginLossWithSampler
from catalyst.contrib.models import MnistSimpleNet
from catalyst.data.sampler import BatchBalanceClassSampler
from catalyst.metrics import AccuracyMetric
from tests import DATA_ROOT

NUM_CLASSES = 4
NUM_FEATURES = 100
NUM_SAMPLES = 200


class DummyModel(nn.Module):
    """Dummy model"""

    def __init__(self, num_features: int, num_classes: int) -> None:
        super().__init__()
        self.model = nn.Sequential(
            nn.Flatten(), nn.Linear(in_features=num_features, out_features=num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward

        Args:
            x: inputs

        Returns:
            model's output
        """
        return self.model(x)


class MnistReIDQGDataset(MnistQGDataset):
    """MnistQGDataset with dummy cids just to test reid pipeline with small dataset"""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._cids = np.random.randint(0, 10, size=len(self._mnist.targets))

    def __getitem__(self, item: int) -> Dict[str, torch.Tensor]:
        """
        Get item

        Args:
            item: item to get

        Returns:
            dict of image, target, cid and is_query key
        """
        sample = super().__getitem__(idx=item)
        sample["cids"] = self._cids[item]
        return sample


class ReIDCustomRunner(dl.SupervisedRunner):
    """ReidCustomRunner for reid case"""

    def handle_batch(self, batch: Dict[str, torch.Tensor]) -> None:
        """
        Process batch

        Args:
            batch: batch data
        """
        if self.is_train_loader:
            images, targets = batch["features"].float(), batch["targets"].long()
            features = self.model(images)
            self.batch = {
                "embeddings": features,
                "targets": targets,
            }
        else:
            images, targets, cids, is_query = (
                batch["features"].float(),
                batch["targets"].long(),
                batch["cids"].long(),
                batch["is_query"].bool(),
            )
            features = self.model(images)
            self.batch = {
                "embeddings": features,
                "targets": targets,
                "cids": cids,
                "is_query": is_query,
            }


@pytest.mark.parametrize(
    "input_key,target_key,keys",
    (
        (
            "inputs_test",
            "logits_test",
            {"inputs_test": "inputs_test", "logits_test": "logits_test"},
        ),
        (
            ["test_1", "test_2", "test_3"],
            ["test_4"],
            {
                "test_1": "test_1",
                "test_2": "test_2",
                "test_3": "test_3",
                "test_4": "test_4",
            },
        ),
        (
            {"test_1": "test_2", "test_3": "test_4"},
            ["test_5"],
            {"test_1": "test_2", "test_3": "test_4", "test_5": "test_5"},
        ),
        (
            {"test_1": "test_2", "test_3": "test_4"},
            {"test_5": "test_6", "test_7": "test_8"},
            {
                "test_1": "test_2",
                "test_3": "test_4",
                "test_5": "test_6",
                "test_7": "test_8",
            },
        ),
    ),
)
def test_format_keys(
    input_key: Union[str, Iterable[str], Dict[str, str]],
    target_key: Union[str, Iterable[str], Dict[str, str]],
    keys: Dict[str, str],
) -> None:
    """Check MetricCallback converts keys correctly"""
    accuracy = AccuracyMetric()
    callback = dl.BatchMetricCallback(
        metric=accuracy, input_key=input_key, target_key=target_key
    )
    assert callback._keys == keys


def test_classification_pipeline():
    """
    Test if classification pipeline can run and compute metrics.
    In this test we check that BatchMetricCallback works with
    AccuracyMetric (ICallbackBatchMetric).
    """
    x = torch.rand(NUM_SAMPLES, NUM_FEATURES)
    y = (torch.rand(NUM_SAMPLES) * NUM_CLASSES).long()
    dataset = TensorDataset(x, y)
    loader = DataLoader(dataset, batch_size=64, num_workers=1)

    model = DummyModel(num_features=NUM_FEATURES, num_classes=NUM_CLASSES)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters())

    runner = dl.SupervisedRunner(
        input_key="features", output_key="logits", target_key="targets"
    )
    with TemporaryDirectory() as logdir:
        runner.train(
            model=model,
            criterion=criterion,
            optimizer=optimizer,
            loaders=OrderedDict({"train": loader, "valid": loader}),
            logdir=logdir,
            num_epochs=3,
            verbose=False,
            valid_loader="valid",
            valid_metric="loss",
            minimize_valid_metric=True,
            callbacks=OrderedDict(
                {
                    "classification": dl.BatchMetricCallback(
                        metric=AccuracyMetric(num_classes=NUM_CLASSES),
                        input_key="logits",
                        target_key="targets",
                    ),
                }
            ),
        )
        assert "accuracy01" in runner.batch_metrics
        assert "accuracy01" in runner.loader_metrics


class CustomRunner(dl.SupervisedRunner):
    """Custom runner for metric learning pipeline"""

    def handle_batch(self, batch: Dict[str, torch.Tensor]) -> None:
        """
        Handle batch for train and valid loaders

        Args:
            batch: batch to process
        """
        if self.is_train_loader:
            images, targets = batch["features"].float(), batch["targets"].long()
            features = self.model(images)
            self.batch = {
                "embeddings": features,
                "targets": targets,
                "images": images,
            }
        else:
            images, targets, is_query = (
                batch["features"].float(),
                batch["targets"].long(),
                batch["is_query"].bool(),
            )
            features = self.model(images)
            self.batch = {
                "embeddings": features,
                "targets": targets,
                "is_query": is_query,
            }


def test_metric_learning_pipeline():
    """
    Test if classification pipeline can run and compute metrics.
    In this test we check that LoaderMetricCallback works with
    CMCMetric (ICallbackLoaderMetric).
    """
    with TemporaryDirectory() as tmp_dir:
        dataset_train = MnistMLDataset(root=tmp_dir, download=True)
        sampler = BatchBalanceClassSampler(
            labels=dataset_train.get_labels(),
            num_classes=3,
            num_samples=10,
            num_batches=10,
        )
        train_loader = DataLoader(
            dataset=dataset_train, batch_sampler=sampler, num_workers=0
        )
        dataset_val = MnistQGDataset(root=tmp_dir, gallery_fraq=0.2)
        val_loader = DataLoader(dataset=dataset_val, batch_size=1024)

        model = DummyModel(num_features=28 * 28, num_classes=NUM_CLASSES)
        optimizer = Adam(model.parameters(), lr=0.001)

        sampler_inbatch = HardTripletsSampler(norm_required=False)
        criterion = TripletMarginLossWithSampler(
            margin=0.5, sampler_inbatch=sampler_inbatch
        )

        callbacks = OrderedDict(
            {
                "cmc": dl.ControlFlowCallbackWrapper(
                    dl.CMCScoreCallback(
                        embeddings_key="embeddings",
                        labels_key="targets",
                        is_query_key="is_query",
                        topk=[1],
                    ),
                    loaders="valid",
                ),
                "control": dl.PeriodicLoaderCallback(
                    valid_loader_key="valid",
                    valid_metric_key="cmc",
                    minimize=False,
                    valid=2,
                ),
            }
        )

        runner = CustomRunner(input_key="features", output_key="embeddings")
        runner.train(
            model=model,
            criterion=criterion,
            optimizer=optimizer,
            callbacks=callbacks,
            loaders=OrderedDict({"train": train_loader, "valid": val_loader}),
            verbose=False,
            valid_loader="valid",
            num_epochs=4,
        )
        assert "cmc01" in runner.loader_metrics


def test_reid_pipeline():
    """This test checks that reid pipeline runs and compute metrics with ReidCMCScoreCallback"""
    with TemporaryDirectory() as logdir:

        # 1. train and valid loaders
        train_dataset = MnistMLDataset(root=DATA_ROOT)
        sampler = BatchBalanceClassSampler(
            labels=train_dataset.get_labels(),
            num_classes=3,
            num_samples=10,
            num_batches=20,
        )
        train_loader = DataLoader(
            dataset=train_dataset, batch_sampler=sampler, num_workers=0
        )

        valid_dataset = MnistReIDQGDataset(root=DATA_ROOT, gallery_fraq=0.2)
        valid_loader = DataLoader(dataset=valid_dataset, batch_size=1024)

        # 2. model and optimizer
        model = MnistSimpleNet(out_features=16)
        optimizer = Adam(model.parameters(), lr=0.001)

        # 3. criterion with triplets sampling
        sampler_inbatch = AllTripletsSampler(max_output_triplets=1000)
        criterion = TripletMarginLossWithSampler(
            margin=0.5, sampler_inbatch=sampler_inbatch
        )

        # 4. training with catalyst Runner
        callbacks = [
            dl.ControlFlowCallbackWrapper(
                dl.CriterionCallback(
                    input_key="embeddings", target_key="targets", metric_key="loss"
                ),
                loaders="train",
            ),
            dl.ControlFlowCallbackWrapper(
                dl.ReidCMCScoreCallback(
                    embeddings_key="embeddings",
                    pids_key="targets",
                    cids_key="cids",
                    is_query_key="is_query",
                    topk=[1],
                ),
                loaders="valid",
            ),
            dl.PeriodicLoaderCallback(
                valid_loader_key="valid",
                valid_metric_key="cmc01",
                minimize=False,
                valid=2,
            ),
        ]

        runner = ReIDCustomRunner()
        runner.train(
            model=model,
            criterion=criterion,
            optimizer=optimizer,
            callbacks=callbacks,
            loaders=OrderedDict({"train": train_loader, "valid": valid_loader}),
            verbose=False,
            logdir=logdir,
            valid_loader="valid",
            valid_metric="cmc01",
            minimize_valid_metric=False,
            num_epochs=10,
        )
        assert "cmc01" in runner.loader_metrics
        assert runner.loader_metrics["cmc01"] > 0.65
