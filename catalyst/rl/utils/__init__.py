# flake8: noqa

from .criterion import *

from .buffers import *
from .gamma import *
from .samplers import *
from .torch import *

from catalyst.utils import *


__all__ = [
    # criterion
    "categorical_loss", "quantile_loss",
    # buffers
    "OffpolicyReplayBuffer", "OnpolicyRolloutBuffer",
    # gamma
    "hyperbolic_gammas",
    # samplers
    "OffpolicyReplaySampler", "OnpolicyRolloutSampler",
    # torch
    "get_trainer_components", "get_network_weights", "set_network_weights",

    ### catalyst.utils
    # argparse
    "args_are_not_none", "boolean_flag",
    # checkpoint
    "pack_checkpoint", "unpack_checkpoint",
    "save_checkpoint", "load_checkpoint",
    # compression
    "compress", "compress_if_needed", "decompress", "decompress_if_needed",
    "pack", "unpack",
    # config
    "load_ordered_yaml", "dump_config",
    "parse_config_args", "parse_args_uargs",
    # dataset
    # ddp
    "is_wrapped_with_ddp", "get_real_module",
    # frozen
    # hash
    "get_hash", "get_short_hash",
    # image
    "imread", "tensor_from_rgb_image", "tensor_to_ndimage",
    "binary_mask_to_overlay_image",
    # initialization
    "create_optimal_inner_init", "outer_init",
    "constant_init", "uniform_init", "normal_init",
    "xavier_init", "kaiming_init", "bias_init_with_prob",
    # misc
    "pairwise", "make_tuple", "merge_dicts", "append_dict",
    # numpy
    "np_softmax", "geometric_cumsum",
    # pandas
    # plotly
    "plot_tensorboard_log",
    # registry
    # seed
    "set_global_seed",
    # serialization
    "serialize", "deserialize",
    # tensorboard
    # torch
    "ce_with_logits", "log1p_exp",
    "normal_sample", "normal_logprob",
    "soft_update",
    "get_optimizable_params",
    "get_optimizer_momentum", "set_optimizer_momentum",
    "assert_fp16_available", "get_device", "get_activation_fn",
    # visualization
    "plot_confusion_matrix", "render_figure_to_tensor"
]
