# flake8: noqa
from catalyst.settings import SETTINGS

if SETTINGS.yaml_required:
    from catalyst.utils.config import save_config, load_config
from catalyst.utils.distributed import (
    get_backend,
    get_world_size,
    get_rank,
    get_nn_from_ddp_module,
    sum_reduce,
    mean_reduce,
    all_gather,
    ddp_reduce,
)

from catalyst.utils.misc import (
    get_utcnow_time,
    maybe_recursive_call,
    get_attr,
    set_global_seed,
    boolean_flag,
    merge_dicts,
    flatten_dict,
    get_hash,
    get_short_hash,
    make_tuple,
    pairwise,
    get_by_keys,
)

from catalyst.utils.onnx import onnx_export

if SETTINGS.onnx_required:
    from catalyst.utils.onnx import quantize_onnx_model

if SETTINGS.pruning_required:
    from catalyst.utils.pruning import prune_model, remove_reparametrization

if SETTINGS.quantization_required:
    from catalyst.utils.quantization import quantize_model

from catalyst.utils.torch import (
    get_optimizer_momentum,
    get_optimizer_momentum_list,
    set_optimizer_momentum,
    get_device,
    get_available_gpus,
    get_available_engine,
    any2device,
    prepare_cudnn,
    get_requires_grad,
    set_requires_grad,
    pack_checkpoint,
    unpack_checkpoint,
    save_checkpoint,
    load_checkpoint,
    soft_update,
    mixup_batch,
)

from catalyst.utils.tracing import trace_model


# from catalyst.contrib.utils import *
