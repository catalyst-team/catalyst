# flake8: noqa
"""
All utils are gathered in :py:mod:`catalyst.utils` for easier access.

.. note::
    Everything from :py:mod:`catalyst.contrib.utils` is included in :py:mod:`catalyst.utils`
"""

from catalyst.contrib.utils import *
from catalyst.tools.settings import IS_PRUNING_AVAILABLE
from catalyst.utils.checkpoint import (
    load_checkpoint,
    pack_checkpoint,
    save_checkpoint,
    unpack_checkpoint,
)
from catalyst.utils.components import process_components
from catalyst.utils.config import load_config, save_config
from catalyst.utils.dict import (
    flatten_dict,
    get_dictkey_auto_fn,
    get_key_all,
    get_key_dict,
    get_key_list,
    get_key_none,
    get_key_str,
    merge_dicts,
    split_dict_to_subdicts,
)
from catalyst.utils.distributed import (
    assert_fp16_available,
    check_apex_available,
    check_ddp_wrapped,
    check_slurm_available,
    check_torch_distributed_initialized,
    get_distributed_env,
    get_distributed_mean,
    get_distributed_params,
    get_nn_from_ddp_module,
    get_rank,
    get_slurm_params,
    initialize_apex,
    is_apex_available,
    is_slurm_available,
    is_torch_distributed_initialized,
    is_wrapped_with_ddp,
)
from catalyst.utils.hash import get_hash, get_short_hash
from catalyst.utils.initialization import (
    get_optimal_inner_init,
    outer_init,
    reset_weights_if_possible,
)
from catalyst.utils.loader import (
    get_native_batch_from_loader,
    get_native_batch_from_loaders,
)
from catalyst.utils.misc import (
    copy_directory,
    fn_ends_with_pass,
    format_metric,
    get_fn_argsnames,
    get_fn_default_params,
    get_utcnow_time,
    is_exception,
    maybe_recursive_call,
)
from catalyst.utils.numpy import get_one_hot
from catalyst.utils.parser import parse_args_uargs, parse_config_args
from catalyst.utils.scripts import (
    distributed_cmd_run,
    dump_base_experiment_code,
    dump_code,
    dump_python_files,
    import_experiment_and_runner,
    import_module,
)
from catalyst.utils.seed import set_global_seed
from catalyst.utils.sys import (
    dump_environment,
    get_environment_vars,
    list_conda_packages,
    list_pip_packages,
)
from catalyst.utils.torch import (
    any2device,
    detach,
    get_activation_fn,
    get_available_gpus,
    get_device,
    get_network_output,
    get_optimizable_params,
    get_optimizer_momentum,
    get_requires_grad,
    prepare_cudnn,
    process_model_params,
    set_optimizer_momentum,
    set_requires_grad,
    trim_tensors,
)

if IS_PRUNING_AVAILABLE:
    from catalyst.utils.pruning import prune_model, remove_reparametrization

from catalyst.tools.settings import IS_GIT_AVAILABLE

if IS_GIT_AVAILABLE:
    from catalyst.utils.pipelines import clone_pipeline
