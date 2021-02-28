# flake8: noqa
import pytest
import torch

from catalyst.metrics.functional._misc import (
    check_consistent_length,
    get_binary_statistics,
    get_multiclass_statistics,
    get_multilabel_statistics,
)


@pytest.mark.parametrize(
    ["outputs", "targets", "tn_true", "fp_true", "fn_true", "tp_true", "support_true"],
    [
        pytest.param(
            torch.tensor([[0, 0, 1, 1, 0, 1, 0, 1]]),
            torch.tensor([[0, 1, 0, 1, 0, 0, 1, 1]]),
            2,
            2,
            2,
            2,
            4,
        ),
    ],
)
def test_get_binary_statistics(
    outputs, targets, tn_true, fp_true, fn_true, tp_true, support_true,
):
    tn, fp, fn, tp, support = get_binary_statistics(outputs, targets)

    assert tn.item() == tn_true
    assert fp.item() == fp_true
    assert fn.item() == fn_true
    assert tp.item() == tp_true
    assert support.item() == support_true


@pytest.mark.parametrize(
    ["outputs", "targets", "tn_true", "fp_true", "fn_true", "tp_true", "support_true"],
    [
        pytest.param(
            torch.tensor([[0, 1, 2, 3]]),
            torch.tensor([[0, 1, 2, 3]]),
            [3.0, 3.0, 3.0, 3.0],
            [0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0],
            [1.0, 1.0, 1.0, 1.0],
            [1.0, 1.0, 1.0, 1.0],
        ),
        pytest.param(
            torch.nn.functional.one_hot(torch.tensor([[0, 1, 2, 3]])),
            torch.tensor([[0, 1, 2, 3]]),
            [3.0, 3.0, 3.0, 3.0],
            [0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0],
            [1.0, 1.0, 1.0, 1.0],
            [1.0, 1.0, 1.0, 1.0],
        ),
        pytest.param(
            torch.tensor([1, 2, 3, 0]),
            torch.tensor([1, 3, 4, 0]),
            [3.0, 3.0, 3.0, 2.0, 3.0],
            [0.0, 0.0, 1.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0, 1.0],
            [1.0, 1.0, 0.0, 0.0, 0.0],
            [1.0, 1.0, 0.0, 1.0, 1.0],
        ),
        pytest.param(
            torch.nn.functional.one_hot(torch.tensor([1, 2, 3, 0])),
            torch.tensor([1, 3, 4, 0]),
            [3.0, 3.0, 3.0, 2.0, 3.0],
            [0.0, 0.0, 1.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0, 1.0],
            [1.0, 1.0, 0.0, 0.0, 0.0],
            [1.0, 1.0, 0.0, 1.0, 1.0],
        ),
    ],
)
def test_get_multiclass_statistics(
    outputs, targets, tn_true, fp_true, fn_true, tp_true, support_true,
):
    tn, fp, fn, tp, support = get_multiclass_statistics(outputs, targets)

    assert torch.allclose(torch.tensor(tn_true).to(tn), tn)
    assert torch.allclose(torch.tensor(fp_true).to(fp), fp)
    assert torch.allclose(torch.tensor(fn_true).to(fn), fn)
    assert torch.allclose(torch.tensor(tp_true).to(tp), tp)
    assert torch.allclose(torch.tensor(support_true).to(support), support)


@pytest.mark.parametrize(
    ["outputs", "targets", "tn_true", "fp_true", "fn_true", "tp_true", "support_true"],
    [
        pytest.param(
            torch.tensor([[0, 0, 1, 1], [0, 1, 0, 1]]),
            torch.tensor([[0, 1, 0, 1], [0, 0, 1, 1]]),
            [2.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 1.0, 0.0],
            [0.0, 1.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 2.0],
            [0.0, 1.0, 1.0, 2.0],
        ),
        pytest.param(
            torch.tensor([[1, 0, 0], [0, 1, 0], [0, 0, 1]]),
            torch.tensor([0, 1, 2]),
            [2.0, 2.0, 2.0],
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
            [1.0, 1.0, 1.0],
            [1.0, 1.0, 1.0],
        ),
        pytest.param(
            torch.tensor([[1, 0, 0], [0, 1, 0], [0, 0, 1]]),
            torch.nn.functional.one_hot(torch.tensor([0, 1, 2])),
            [2.0, 2.0, 2.0],
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
            [1.0, 1.0, 1.0],
            [1.0, 1.0, 1.0],
        ),
    ],
)
def test_get_multilabel_statistics(
    outputs, targets, tn_true, fp_true, fn_true, tp_true, support_true,
):
    tn, fp, fn, tp, support = get_multilabel_statistics(outputs, targets)

    assert torch.allclose(torch.tensor(tn_true).to(tn), tn)
    assert torch.allclose(torch.tensor(fp_true).to(fp), fp)
    assert torch.allclose(torch.tensor(fn_true).to(fn), fn)
    assert torch.allclose(torch.tensor(tp_true).to(tp), tp)
    assert torch.allclose(torch.tensor(support_true).to(support), support)


@pytest.mark.parametrize(
    ["outputs", "targets"],
    [
        pytest.param(
            torch.tensor([[4.0, 2.0, 3.0, 1.0], [1.0, 2.0, 3.0, 4.0]]),
            torch.tensor([[0, 0, 1.0, 1.0, 1.0], [0, 0, 1.0, 1.0, 1.0]]),
        ),
    ],
)
def test_check_consistent_length(outputs, targets):
    with pytest.raises(ValueError) as execinfo:
        check_consistent_length(outputs, targets)
    assert str(execinfo.value) == "Inconsistent numbers of samples"
