# flake8: noqa

from catalyst.settings import IS_PRUNING_AVAILABLE, IS_QUANTIZATION_AVAILABLE, IS_ONNX_AVAILABLE

from catalyst.utils.checkpoint import (
    load_checkpoint,
    pack_checkpoint,
    save_checkpoint,
    unpack_checkpoint,
)

from catalyst.utils.config import load_config, save_config

from catalyst.utils.data import get_loaders_from_params, get_loader

from catalyst.utils.distributed import (
    get_nn_from_ddp_module,
    get_slurm_params,
    get_distributed_params,
    get_distributed_env,
    get_rank,
    check_ddp_wrapped,
    check_torch_distributed_initialized,
    check_slurm_available,
    check_apex_available,
    check_amp_available,
    assert_fp16_available,
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

if IS_ONNX_AVAILABLE:
    from catalyst.utils.onnx import quantize_onnx_model, convert_to_onnx

if IS_PRUNING_AVAILABLE:
    from catalyst.utils.pruning import prune_model, remove_reparametrization

if IS_QUANTIZATION_AVAILABLE:
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
    any2device,
    prepare_cudnn,
    process_model_params,
    get_requires_grad,
    set_requires_grad,
    get_network_output,
    detach,
    trim_tensors,
    get_optimal_inner_init,
    outer_init,
    reset_weights_if_possible,
)

from catalyst.utils.tracing import trace_model


from catalyst.contrib.utils import *
