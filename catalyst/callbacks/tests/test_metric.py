from typing import Union, Iterable, Dict

import pytest
from catalyst.callbacks.metric import BatchMetricCallback
from catalyst.metrics import AccuracyMetric


@pytest.mark.parametrize(
    "input_key,target_key,keys",
    (
        (
            "inputs_test",
            "logits_test",
            {"inputs_test": "inputs_test", "logits_test": "logits_test", },
        ),
        (
            ["test_1", "test_2", "test_3", ],
            ["test_4", ],
            {"test_1": "test_1", "test_2": "test_2", "test_3": "test_3", "test_4": "test_4", },
        ),
        (
            {"test_1": "test_2", "test_3": "test_4", },
            ["test_5", ],
            {"test_1": "test_2", "test_3": "test_4", "test_5": "test_5", },
        ),
        (
            {"test_1": "test_2", "test_3": "test_4", },
            {"test_5": "test_6", "test_7": "test_8", },
            {"test_1": "test_2", "test_3": "test_4", "test_5": "test_6", "test_7": "test_8", },
        )
    )
)
def test_format_keys(
        input_key: Union[str, Iterable[str], Dict[str, str]],
        target_key: Union[str, Iterable[str], Dict[str, str]],
        keys: Dict[str, str],
) -> None:
    """Check MetricCallback converts keys correctly"""
    accuracy = AccuracyMetric()
    callback = BatchMetricCallback(
       metric=accuracy, input_key=input_key, target_key=target_key
    )
    assert callback._keys == keys
