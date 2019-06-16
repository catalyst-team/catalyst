import numpy as np

from catalyst.dl.core.metric_manager import MetricManager


def test_to_value():
    MetricManager._to_single_value(np.float32(1.0))


def test_epoch_metrics():
    metrics = MetricManager("valid", "test", True)

    metrics.begin_epoch()
    metrics.begin_loader("train")

    metrics.begin_batch()
    metrics.add_batch_value("test", 2)
    metrics.end_batch()

    metrics.begin_batch()
    metrics.add_batch_value("test", 2)
    metrics.end_batch()

    metrics.end_loader()
    metrics.begin_loader("valid")

    metrics.begin_batch()
    metrics.add_batch_value("test", 1)
    metrics.end_batch()

    metrics.begin_batch()
    metrics.add_batch_value("test", 0)
    metrics.end_batch()

    metrics.end_loader()
    metrics.end_epoch_train()

    assert metrics.epoch_values["valid"]["test"] == 0.5
    assert metrics.epoch_values["train"]["test"] == 2


def test_best():
    metrics = MetricManager("valid", "test", True)

    metrics.begin_epoch()
    metrics.begin_loader("valid")

    metrics.begin_batch()
    metrics.add_batch_value("test", 1)
    metrics.end_batch()

    metrics.end_loader()
    metrics.end_epoch_train()

    metrics.begin_epoch()
    metrics.begin_loader("valid")

    metrics.begin_batch()
    metrics.add_batch_value("test", 0)
    metrics.end_batch()

    metrics.end_loader()
    metrics.end_epoch_train()

    assert metrics.is_best
    assert metrics.best_main_metric_value == 0

    metrics = MetricManager("valid", "test", False)

    metrics.begin_epoch()
    metrics.begin_loader("valid")

    metrics.begin_batch()
    metrics.add_batch_value("test", 1)
    metrics.end_batch()

    metrics.end_loader()
    metrics.end_epoch_train()

    metrics.begin_epoch()
    metrics.begin_loader("valid")

    metrics.begin_batch()
    metrics.add_batch_value("test", 0)
    metrics.end_batch()

    metrics.end_loader()
    metrics.end_epoch_train()

    assert not metrics.is_best
    assert metrics.best_main_metric_value == 1
