from unittest.mock import MagicMock, PropertyMock

from catalyst.core import EarlyStoppingCallback


def test_patience1():
    """@TODO: Docs. Contribution is welcome."""
    early_stop = EarlyStoppingCallback(1)
    runner = MagicMock()
    type(runner).stage_name = PropertyMock(return_value="training")
    type(runner).valid_metrics = PropertyMock(return_value={"loss": 0.001})
    stop_mock = PropertyMock(return_value=False)
    type(runner).need_early_stop = stop_mock

    early_stop.on_epoch_end(runner)

    assert stop_mock.mock_calls == []
