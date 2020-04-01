from torch import nn

from catalyst import utils


def test_get_fn_argsnames():
    """@TODO: Docs. Contribution is welcome."""

    class Net1(nn.Module):
        def forward(self, x):
            return x

    class Net2(nn.Module):
        def forward(self, x, y):
            return x

    class Net3(nn.Module):
        def forward(self, x, y=None):
            return x

    class Net4(nn.Module):
        def forward(self, x, *, y=None):
            return x

    class Net5(nn.Module):
        def forward(self, *, x):
            return x

    class Net6(nn.Module):
        def forward(self, *, x, y):
            return x

    class Net7(nn.Module):
        def forward(self, *, x, y=None):
            return x

    nets = [Net1, Net2, Net3, Net4, Net5, Net6, Net7]
    params_true = [
        ["x"],
        ["x", "y"],
        ["x", "y"],
        ["x", "y"],
        ["x"],
        ["x", "y"],
        ["x", "y"],
    ]

    params_predicted = list(
        map(
            lambda x: utils.get_fn_argsnames(x.forward, exclude=["self"]), nets
        )
    )
    assert params_predicted == params_true


def test_fn_ends_with_pass():
    """@TODO: Docs. Contribution is welcome."""

    def useless_fn():
        pass

    def usefull_fn():
        print("I am useful!")

    assert utils.fn_ends_with_pass(useless_fn) is True
    assert utils.fn_ends_with_pass(usefull_fn) is False


def test_fn_ends_with_pass_on_callbacks():
    """@TODO: Docs. Contribution is welcome."""

    def test_fn_ends_with_pass_on_callback(
        callback, events,
    ):
        for event in events["covered"]:
            fn_name = f"on_{event}"
            assert (
                utils.fn_ends_with_pass(getattr(callback.__class__, fn_name))
                is False
            )
        for event in events["non-covered"]:
            fn_name = f"on_{event}"
            assert (
                utils.fn_ends_with_pass(getattr(callback.__class__, fn_name))
                is True
            )

    # Callback test
    from catalyst.dl import Callback

    callback = Callback(order=1)
    start_events = [
        "stage_start",
        "epoch_start",
        "batch_start",
        "loader_start",
    ]
    end_events = [
        "stage_end",
        "epoch_end",
        "batch_end",
        "loader_end",
        "exception",
    ]
    events = {"covered": [], "non-covered": [*start_events, *end_events]}
    test_fn_ends_with_pass_on_callback(callback=callback, events=events)

    # CriterionCallback test
    from catalyst.dl import CriterionCallback

    callback = CriterionCallback()
    covered_events = ["stage_start", "batch_end"]
    non_covered_start_events = ["epoch_start", "batch_start", "loader_start"]
    non_covered_end_events = [
        "stage_end",
        "epoch_end",
        "loader_end",
        "exception",
    ]
    events = {
        "covered": [*covered_events],
        "non-covered": [*non_covered_start_events, *non_covered_end_events],
    }
    test_fn_ends_with_pass_on_callback(callback=callback, events=events)
