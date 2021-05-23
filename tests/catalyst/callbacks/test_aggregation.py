# flake8: noqa
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from catalyst import dl, utils
from catalyst.contrib.nn.criterion import FocalLossBinary


def prepare_experiment():
    # data
    utils.set_global_seed(42)
    num_samples, num_features, num_classes = int(1e4), int(1e1), 4
    X = torch.rand(num_samples, num_features)
    y = (torch.rand(num_samples,) * num_classes).to(torch.int64)
    y = torch.nn.functional.one_hot(y, num_classes).double()

    # pytorch loaders
    dataset = TensorDataset(X, y)
    loader = DataLoader(dataset, batch_size=32, num_workers=1)
    loaders = {"train": loader, "valid": loader}

    # model, criterion, optimizer
    model = torch.nn.Linear(num_features, num_classes)
    criterion = {"bce": torch.nn.BCEWithLogitsLoss(), "focal": FocalLossBinary()}
    optimizer = torch.optim.Adam(model.parameters())
    return loaders, model, criterion, optimizer


def test_aggregation_1():
    """
    Aggregation as weighted_sum
    """
    loaders, model, criterion, optimizer = prepare_experiment()
    runner = dl.SupervisedRunner()
    runner.train(
        model=model,
        criterion=criterion,
        optimizer=optimizer,
        loaders=loaders,
        logdir="./logs/aggregation_1/",
        num_epochs=3,
        callbacks=[
            dl.CriterionCallback(
                input_key="logits",
                target_key="targets",
                metric_key="loss_bce",
                criterion_key="bce",
            ),
            dl.CriterionCallback(
                input_key="logits",
                target_key="targets",
                metric_key="loss_focal",
                criterion_key="focal",
            ),
            # loss aggregation
            dl.MetricAggregationCallback(
                metric_key="loss",
                metrics={"loss_focal": 0.6, "loss_bce": 0.4},
                mode="weighted_sum",
            ),
        ],
    )
    for loader in ["train", "valid"]:
        metrics = runner.epoch_metrics[loader]
        loss_1 = metrics["loss_bce"] * 0.4 + metrics["loss_focal"] * 0.6
        loss_2 = metrics["loss"]
        assert np.abs(loss_1 - loss_2) < 1e-5


def test_aggregation_2():
    """
    Aggregation with custom function
    """
    loaders, model, criterion, optimizer = prepare_experiment()
    runner = dl.SupervisedRunner()

    def aggregation_function(metrics, runner):
        epoch = runner.stage_epoch_step
        loss = (3 / 2 - epoch / 2) * metrics["loss_focal"] + (1 / 2 * epoch - 1 / 2) * metrics[
            "loss_bce"
        ]
        return loss

    runner.train(
        model=model,
        criterion=criterion,
        optimizer=optimizer,
        loaders=loaders,
        logdir="./logs/aggregation_2/",
        num_epochs=3,
        callbacks=[
            dl.CriterionCallback(
                input_key="logits",
                target_key="targets",
                metric_key="loss_bce",
                criterion_key="bce",
            ),
            dl.CriterionCallback(
                input_key="logits",
                target_key="targets",
                metric_key="loss_focal",
                criterion_key="focal",
            ),
            # loss aggregation
            dl.MetricAggregationCallback(metric_key="loss", mode=aggregation_function),
        ],
    )
    for loader in ["train", "valid"]:
        metrics = runner.epoch_metrics[loader]
        loss_1 = metrics["loss_bce"]
        loss_2 = metrics["loss"]
        assert np.abs(loss_1 - loss_2) < 1e-5
