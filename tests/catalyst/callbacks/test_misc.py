# flake8: noqa
from unittest.mock import MagicMock, PropertyMock

from catalyst.callbacks import EarlyStoppingCallback


def test_patience1():
    """Tests EarlyStoppingCallback."""
    early_stop = EarlyStoppingCallback(
        patience=1, loader_key="valid", metric_key="loss", minimize=True
    )
    runner = MagicMock()
    type(runner).stage_key = PropertyMock(return_value="training")
    type(runner).epoch_metrics = PropertyMock(return_value={"valid": {"loss": 0.001}})
    stop_mock = PropertyMock(return_value=False)
    type(runner).need_early_stop = stop_mock

    early_stop.on_epoch_end(runner)

    assert stop_mock.mock_calls == []
