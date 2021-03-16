# flake8: noqa
from catalyst.settings import SETTINGS

from catalyst.utils.config import load_config, save_config

from catalyst.utils.data import get_loaders_from_params, get_loader

from catalyst.utils.distributed import (
    get_distributed_params,
    get_rank,
    get_nn_from_ddp_module,
    sum_reduce,
    mean_reduce,
    all_gather,
)

from catalyst.utils.misc import (
    get_fn_default_params,
    get_fn_argsnames,
    get_utcnow_time,
    is_exception,
    maybe_recursive_call,
    get_attr,
    set_global_seed,
    boolean_flag,
    get_dictkey_auto_fn,
    merge_dicts,
    flatten_dict,
    get_hash,
    get_short_hash,
    args_are_not_none,
    make_tuple,
    pairwise,
    find_value_ids,
    get_by_keys,
    convert_labels2list,
)
from catalyst.utils.numpy import get_one_hot

from catalyst.utils.onnx import onnx_export

if SETTINGS.onnx_required:
    from catalyst.utils.onnx import quantize_onnx_model

if SETTINGS.pruning_required:
    from catalyst.utils.pruning import prune_model, remove_reparametrization

if SETTINGS.quantization_required:
    from catalyst.utils.quantization import quantize_model

from catalyst.utils.swa import (
    average_weights,
    get_averaged_weights_by_path_mask,
)

from catalyst.utils.sys import (
    import_module,
    dump_code,
    dump_environment,
    get_config_runner,
)

from catalyst.utils.torch import (
    get_optimizable_params,
    get_optimizer_momentum,
    get_optimizer_momentum_list,
    set_optimizer_momentum,
    get_device,
    get_available_gpus,
    get_available_engine,
    any2device,
    prepare_cudnn,
    process_model_params,
    get_requires_grad,
    set_requires_grad,
    get_network_output,
    detach_tensor,
    trim_tensors,
    get_optimal_inner_init,
    outer_init,
    reset_weights_if_possible,
    pack_checkpoint,
    unpack_checkpoint,
    save_checkpoint,
    load_checkpoint,
)

from catalyst.utils.tracing import trace_model


from catalyst.contrib.utils import *
