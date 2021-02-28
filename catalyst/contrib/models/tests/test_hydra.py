from collections import OrderedDict
import copy
from pathlib import Path

import pytest
import torch
from torch import nn

from catalyst import utils
from catalyst.contrib.models import Hydra, SequentialNet
from catalyst.contrib.nn.modules import Normalize


def _pop_normalization(dct):
    for values in dct.values():
        if isinstance(values, dict):
            values.pop("normalize_output", None)
            _pop_normalization(values)


def _check_lists(left, right):
    assert sorted(left) == sorted(right)


def _check_named_parameters(left, right):
    left_keys = dict(left.named_parameters()).keys()
    right_keys = dict(right.named_parameters()).keys()
    _check_lists(left_keys, right_keys)


def test_config1():
    """@TODO: Docs. Contribution is welcome."""
    config1 = {
        "encoder_params": {
            "hiddens": [16, 16],
            "layer_fn": {"module": "Linear", "bias": False},
            "norm_fn": "LayerNorm",
        },
        "heads_params": {
            "head1": {"hiddens": [2], "layer_fn": {"module": "Linear", "bias": True}},
            "_head2": {
                "_hidden": {"hiddens": [16], "layer_fn": {"module": "Linear", "bias": False}},
                "head2_1": {
                    "hiddens": [32],
                    "layer_fn": {"module": "Linear", "bias": True},
                    "normalize_output": True,
                },
                "_head2_2": {
                    "_hidden": {
                        "hiddens": [16, 16, 16],
                        "layer_fn": {"module": "Linear", "bias": False},
                    },
                    "head2_2_1": {
                        "hiddens": [32],
                        "layer_fn": {"module": "Linear", "bias": True},
                        "normalize_output": False,
                    },
                },
            },
        },
        "embedders_params": {
            "target1": {"num_embeddings": 2, "normalize_output": True},
            "target2": {"num_embeddings": 2, "normalize_output": False},
        },
    }

    hydra = Hydra.get_from_params(**config1)

    config1_copy = copy.deepcopy(config1)
    _pop_normalization(config1_copy)
    encoder_params = config1_copy["encoder_params"]
    heads_params = config1_copy["heads_params"]
    heads_params["head1"]["hiddens"].insert(0, 16)
    heads_params["_head2"]["_hidden"]["hiddens"].insert(0, 16)
    heads_params["_head2"]["head2_1"]["hiddens"].insert(0, 16)
    heads_params["_head2"]["_head2_2"]["_hidden"]["hiddens"].insert(0, 16)
    heads_params["_head2"]["_head2_2"]["head2_2_1"]["hiddens"].insert(0, 16)

    net = nn.ModuleDict(
        {
            "encoder": SequentialNet(**encoder_params),
            "embedders": nn.ModuleDict(
                {
                    "target1": nn.Sequential(
                        OrderedDict(
                            [
                                ("embedding", nn.Embedding(embedding_dim=16, num_embeddings=2)),
                                ("normalize", Normalize()),
                            ]
                        )
                    ),
                    "target2": nn.Sequential(
                        OrderedDict(
                            [("embedding", nn.Embedding(embedding_dim=16, num_embeddings=2))]
                        )
                    ),
                }
            ),
            "heads": nn.ModuleDict(
                {
                    "head1": nn.Sequential(
                        OrderedDict([("net", SequentialNet(**heads_params["head1"]))])
                    ),
                    "_head2": nn.ModuleDict(
                        {
                            "_hidden": nn.Sequential(
                                OrderedDict(
                                    [("net", SequentialNet(**heads_params["_head2"]["_hidden"]))]
                                )
                            ),
                            "head2_1": nn.Sequential(
                                OrderedDict(
                                    [
                                        (
                                            "net",
                                            SequentialNet(**heads_params["_head2"]["head2_1"]),
                                        ),
                                        ("normalize", Normalize()),
                                    ]
                                )
                            ),
                            "_head2_2": nn.ModuleDict(
                                {
                                    "_hidden": nn.Sequential(
                                        OrderedDict(
                                            [
                                                (
                                                    "net",
                                                    SequentialNet(
                                                        **heads_params["_head2"]["_head2_2"][
                                                            "_hidden"
                                                        ]
                                                    ),
                                                )
                                            ]
                                        )
                                    ),
                                    "head2_2_1": nn.Sequential(
                                        OrderedDict(
                                            [
                                                (
                                                    "net",
                                                    SequentialNet(
                                                        **heads_params["_head2"]["_head2_2"][
                                                            "head2_2_1"
                                                        ]
                                                    ),
                                                )
                                            ]
                                        )
                                    ),
                                }
                            ),
                        }
                    ),
                }
            ),
        }
    )

    _check_named_parameters(hydra.encoder, net["encoder"])
    _check_named_parameters(hydra.heads, net["heads"])
    _check_named_parameters(hydra.embedders, net["embedders"])

    input_ = torch.rand(1, 16)

    output_kv = hydra(input_)
    assert (input_ == output_kv["features"]).sum().item() == 16
    kv_keys = [
        "features",
        "embeddings",
        "head1",
        "_head2/",
        "_head2/head2_1",
        "_head2/_head2_2/",
        "_head2/_head2_2/head2_2_1",
    ]
    _check_lists(output_kv.keys(), kv_keys)

    output_kv = hydra(input_, target1=torch.ones(1, 2).long())
    kv_keys = [
        "features",
        "embeddings",
        "head1",
        "_head2/",
        "_head2/head2_1",
        "_head2/_head2_2/",
        "_head2/_head2_2/head2_2_1",
        "target1_embeddings",
    ]
    _check_lists(output_kv.keys(), kv_keys)

    output_kv = hydra(input_, target2=torch.ones(1, 2).long())
    kv_keys = [
        "features",
        "embeddings",
        "head1",
        "_head2/",
        "_head2/head2_1",
        "_head2/_head2_2/",
        "_head2/_head2_2/head2_2_1",
        "target2_embeddings",
    ]
    _check_lists(output_kv.keys(), kv_keys)

    output_kv = hydra(input_, target1=torch.ones(1, 2).long(), target2=torch.ones(1, 2).long())
    kv_keys = [
        "features",
        "embeddings",
        "head1",
        "_head2/",
        "_head2/head2_1",
        "_head2/_head2_2/",
        "_head2/_head2_2/head2_2_1",
        "target1_embeddings",
        "target2_embeddings",
    ]
    _check_lists(output_kv.keys(), kv_keys)

    output_tuple = hydra.forward_tuple(input_)
    assert len(output_tuple) == 5
    assert (output_tuple[0] == output_kv["features"]).sum().item() == 16
    assert (output_tuple[1] == output_kv["embeddings"]).sum().item() == 16


def test_config2():
    """@TODO: Docs. Contribution is welcome."""
    config2 = {
        "in_features": 16,
        "heads_params": {
            "head1": {"hiddens": [2], "layer_fn": {"module": "Linear", "bias": True}},
            "_head2": {
                "_hidden": {"hiddens": [16], "layer_fn": {"module": "Linear", "bias": False}},
                "head2_1": {
                    "hiddens": [32],
                    "layer_fn": {"module": "Linear", "bias": True},
                    "normalize_output": True,
                },
                "_head2_2": {
                    "_hidden": {
                        "hiddens": [16, 16, 16],
                        "layer_fn": {"module": "Linear", "bias": False},
                    },
                    "head2_2_1": {
                        "hiddens": [32],
                        "layer_fn": {"module": "Linear", "bias": True},
                        "normalize_output": False,
                    },
                },
            },
        },
    }

    hydra = Hydra.get_from_params(**config2)

    config2_copy = copy.deepcopy(config2)
    _pop_normalization(config2_copy)
    heads_params = config2_copy["heads_params"]
    heads_params["head1"]["hiddens"].insert(0, 16)
    heads_params["_head2"]["_hidden"]["hiddens"].insert(0, 16)
    heads_params["_head2"]["head2_1"]["hiddens"].insert(0, 16)
    heads_params["_head2"]["_head2_2"]["_hidden"]["hiddens"].insert(0, 16)
    heads_params["_head2"]["_head2_2"]["head2_2_1"]["hiddens"].insert(0, 16)

    net = nn.ModuleDict(
        {
            "encoder": nn.Sequential(),
            "heads": nn.ModuleDict(
                {
                    "head1": nn.Sequential(
                        OrderedDict([("net", SequentialNet(**heads_params["head1"]))])
                    ),
                    "_head2": nn.ModuleDict(
                        {
                            "_hidden": nn.Sequential(
                                OrderedDict(
                                    [("net", SequentialNet(**heads_params["_head2"]["_hidden"]))]
                                )
                            ),
                            "head2_1": nn.Sequential(
                                OrderedDict(
                                    [
                                        (
                                            "net",
                                            SequentialNet(**heads_params["_head2"]["head2_1"]),
                                        ),
                                        ("normalize", Normalize()),
                                    ]
                                )
                            ),
                            "_head2_2": nn.ModuleDict(
                                {
                                    "_hidden": nn.Sequential(
                                        OrderedDict(
                                            [
                                                (
                                                    "net",
                                                    SequentialNet(
                                                        **heads_params["_head2"]["_head2_2"][
                                                            "_hidden"
                                                        ]
                                                    ),
                                                )
                                            ]
                                        )
                                    ),
                                    "head2_2_1": nn.Sequential(
                                        OrderedDict(
                                            [
                                                (
                                                    "net",
                                                    SequentialNet(
                                                        **heads_params["_head2"]["_head2_2"][
                                                            "head2_2_1"
                                                        ]
                                                    ),
                                                )
                                            ]
                                        )
                                    ),
                                }
                            ),
                        }
                    ),
                }
            ),
        }
    )

    _check_named_parameters(hydra.encoder, net["encoder"])
    _check_named_parameters(hydra.heads, net["heads"])
    assert hydra.embedders == {}

    input_ = torch.rand(1, 16)

    output_kv = hydra(input_)
    assert (input_ == output_kv["features"]).sum().item() == 16
    assert (input_ == output_kv["embeddings"]).sum().item() == 16
    kv_keys = [
        "features",
        "embeddings",
        "head1",
        "_head2/",
        "_head2/head2_1",
        "_head2/_head2_2/",
        "_head2/_head2_2/head2_2_1",
    ]
    _check_lists(output_kv.keys(), kv_keys)

    with pytest.raises(KeyError):
        output_kv = hydra(input_, target1=torch.ones(1, 2).long())
    with pytest.raises(KeyError):
        output_kv = hydra(input_, target2=torch.ones(1, 2).long())
    with pytest.raises(KeyError):
        output_kv = hydra(
            input_, target1=torch.ones(1, 2).long(), target2=torch.ones(1, 2).long(),
        )

    output_tuple = hydra.forward_tuple(input_)
    assert len(output_tuple) == 5
    assert (output_tuple[0] == output_kv["features"]).sum().item() == 16
    assert (output_tuple[1] == output_kv["embeddings"]).sum().item() == 16


def test_config3():
    """@TODO: Docs. Contribution is welcome."""
    config_path = Path(__file__).absolute().parent / "config3.yml"
    config3 = utils.load_config(config_path)["model_params"]

    hydra = Hydra.get_from_params(**config3)

    config3_copy = copy.deepcopy(config3)
    _pop_normalization(config3_copy)
    encoder_params = config3_copy["encoder_params"]
    heads_params = config3_copy["heads_params"]
    heads_params["head1"]["hiddens"].insert(0, 16)
    heads_params["_head2"]["_hidden"]["hiddens"].insert(0, 16)
    heads_params["_head2"]["head2_1"]["hiddens"].insert(0, 16)
    heads_params["_head2"]["_head2_2"]["_hidden"]["hiddens"].insert(0, 16)
    heads_params["_head2"]["_head2_2"]["head2_2_1"]["hiddens"].insert(0, 16)

    net = nn.ModuleDict(
        {
            "encoder": SequentialNet(**encoder_params),
            "embedders": nn.ModuleDict(
                {
                    "target1": nn.Sequential(
                        OrderedDict(
                            [
                                ("embedding", nn.Embedding(embedding_dim=16, num_embeddings=2)),
                                ("normalize", Normalize()),
                            ]
                        )
                    ),
                    "target2": nn.Sequential(
                        OrderedDict(
                            [("embedding", nn.Embedding(embedding_dim=16, num_embeddings=2))]
                        )
                    ),
                }
            ),
            "heads": nn.ModuleDict(
                {
                    "head1": nn.Sequential(
                        OrderedDict([("net", SequentialNet(**heads_params["head1"]))])
                    ),
                    "_head2": nn.ModuleDict(
                        {
                            "_hidden": nn.Sequential(
                                OrderedDict(
                                    [("net", SequentialNet(**heads_params["_head2"]["_hidden"]))]
                                )
                            ),
                            "head2_1": nn.Sequential(
                                OrderedDict(
                                    [
                                        (
                                            "net",
                                            SequentialNet(**heads_params["_head2"]["head2_1"]),
                                        ),
                                        ("normalize", Normalize()),
                                    ]
                                )
                            ),
                            "_head2_2": nn.ModuleDict(
                                {
                                    "_hidden": nn.Sequential(
                                        OrderedDict(
                                            [
                                                (
                                                    "net",
                                                    SequentialNet(
                                                        **heads_params["_head2"]["_head2_2"][
                                                            "_hidden"
                                                        ]
                                                    ),
                                                )
                                            ]
                                        )
                                    ),
                                    "head2_2_1": nn.Sequential(
                                        OrderedDict(
                                            [
                                                (
                                                    "net",
                                                    SequentialNet(
                                                        **heads_params["_head2"]["_head2_2"][
                                                            "head2_2_1"
                                                        ]
                                                    ),
                                                )
                                            ]
                                        )
                                    ),
                                }
                            ),
                        }
                    ),
                }
            ),
        }
    )

    _check_named_parameters(hydra.encoder, net["encoder"])
    _check_named_parameters(hydra.heads, net["heads"])
    _check_named_parameters(hydra.embedders, net["embedders"])

    input_ = torch.rand(1, 16)

    output_kv = hydra(input_)
    assert (input_ == output_kv["features"]).sum().item() == 16
    kv_keys = [
        "features",
        "embeddings",
        "head1",
        "_head2/",
        "_head2/head2_1",
        "_head2/_head2_2/",
        "_head2/_head2_2/head2_2_1",
    ]
    _check_lists(output_kv.keys(), kv_keys)

    output_kv = hydra(input_, target1=torch.ones(1, 2).long())
    kv_keys = [
        "features",
        "embeddings",
        "head1",
        "_head2/",
        "_head2/head2_1",
        "_head2/_head2_2/",
        "_head2/_head2_2/head2_2_1",
        "target1_embeddings",
    ]
    _check_lists(output_kv.keys(), kv_keys)

    output_kv = hydra(input_, target2=torch.ones(1, 2).long())
    kv_keys = [
        "features",
        "embeddings",
        "head1",
        "_head2/",
        "_head2/head2_1",
        "_head2/_head2_2/",
        "_head2/_head2_2/head2_2_1",
        "target2_embeddings",
    ]
    _check_lists(output_kv.keys(), kv_keys)

    output_kv = hydra(input_, target1=torch.ones(1, 2).long(), target2=torch.ones(1, 2).long())
    kv_keys = [
        "features",
        "embeddings",
        "head1",
        "_head2/",
        "_head2/head2_1",
        "_head2/_head2_2/",
        "_head2/_head2_2/head2_2_1",
        "target1_embeddings",
        "target2_embeddings",
    ]
    _check_lists(output_kv.keys(), kv_keys)

    output_tuple = hydra.forward_tuple(input_)
    assert len(output_tuple) == 5
    assert (output_tuple[0] == output_kv["features"]).sum().item() == 16
    assert (output_tuple[1] == output_kv["embeddings"]).sum().item() == 16


def test_config4():
    """@TODO: Docs. Contribution is welcome."""
    config_path = Path(__file__).absolute().parent / "config4.yml"
    config4 = utils.load_config(config_path)["model_params"]

    with pytest.raises(AssertionError):
        hydra = Hydra.get_from_params(**config4)
    config4["in_features"] = 16
    hydra = Hydra.get_from_params(**config4)

    config4_copy = copy.deepcopy(config4)
    _pop_normalization(config4_copy)
    heads_params = config4_copy["heads_params"]
    heads_params["head1"]["hiddens"].insert(0, 16)
    heads_params["_head2"]["_hidden"]["hiddens"].insert(0, 16)
    heads_params["_head2"]["head2_1"]["hiddens"].insert(0, 16)
    heads_params["_head2"]["_head2_2"]["_hidden"]["hiddens"].insert(0, 16)
    heads_params["_head2"]["_head2_2"]["head2_2_1"]["hiddens"].insert(0, 16)

    net = nn.ModuleDict(
        {
            "encoder": nn.Sequential(),
            "heads": nn.ModuleDict(
                {
                    "head1": nn.Sequential(
                        OrderedDict([("net", SequentialNet(**heads_params["head1"]))])
                    ),
                    "_head2": nn.ModuleDict(
                        {
                            "_hidden": nn.Sequential(
                                OrderedDict(
                                    [("net", SequentialNet(**heads_params["_head2"]["_hidden"]))]
                                )
                            ),
                            "head2_1": nn.Sequential(
                                OrderedDict(
                                    [
                                        (
                                            "net",
                                            SequentialNet(**heads_params["_head2"]["head2_1"]),
                                        ),
                                        ("normalize", Normalize()),
                                    ]
                                )
                            ),
                            "_head2_2": nn.ModuleDict(
                                {
                                    "_hidden": nn.Sequential(
                                        OrderedDict(
                                            [
                                                (
                                                    "net",
                                                    SequentialNet(
                                                        **heads_params["_head2"]["_head2_2"][
                                                            "_hidden"
                                                        ]
                                                    ),
                                                )
                                            ]
                                        )
                                    ),
                                    "head2_2_1": nn.Sequential(
                                        OrderedDict(
                                            [
                                                (
                                                    "net",
                                                    SequentialNet(
                                                        **heads_params["_head2"]["_head2_2"][
                                                            "head2_2_1"
                                                        ]
                                                    ),
                                                )
                                            ]
                                        )
                                    ),
                                }
                            ),
                        }
                    ),
                }
            ),
        }
    )

    _check_named_parameters(hydra.encoder, net["encoder"])
    _check_named_parameters(hydra.heads, net["heads"])
    assert hydra.embedders == {}

    input_ = torch.rand(1, 16)

    output_kv = hydra(input_)
    assert (input_ == output_kv["features"]).sum().item() == 16
    assert (input_ == output_kv["embeddings"]).sum().item() == 16
    kv_keys = [
        "features",
        "embeddings",
        "head1",
        "_head2/",
        "_head2/head2_1",
        "_head2/_head2_2/",
        "_head2/_head2_2/head2_2_1",
    ]
    _check_lists(output_kv.keys(), kv_keys)

    with pytest.raises(KeyError):
        output_kv = hydra(input_, target1=torch.ones(1, 2).long())
    with pytest.raises(KeyError):
        output_kv = hydra(input_, target2=torch.ones(1, 2).long())
    with pytest.raises(KeyError):
        output_kv = hydra(
            input_, target1=torch.ones(1, 2).long(), target2=torch.ones(1, 2).long(),
        )

    output_tuple = hydra.forward_tuple(input_)
    assert len(output_tuple) == 5
    assert (output_tuple[0] == output_kv["features"]).sum().item() == 16
    assert (output_tuple[1] == output_kv["embeddings"]).sum().item() == 16
