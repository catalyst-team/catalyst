# flake8: noqa
"""
All utils are gathered in :py:mod:`catalyst.utils` for easier access.

.. note::
    Everything from :py:mod:`catalyst.contrib.utils` is included in :py:mod:`catalyst.utils`
"""

from catalyst.contrib.utils import *

from catalyst.utils.checkpoint import (
    load_checkpoint,
    pack_checkpoint,
    save_checkpoint,
    unpack_checkpoint,
)
from catalyst.utils.components import process_components
from catalyst.utils.config import load_config, save_config
from catalyst.utils.dict import (
    get_key_str,
    get_key_none,
    get_key_list,
    get_key_dict,
    get_key_all,
    get_dictkey_auto_fn,
    merge_dicts,
    flatten_dict,
    split_dict_to_subdicts,
)
from catalyst.utils.distributed import (
    get_nn_from_ddp_module,
    get_slurm_params,
    get_distributed_params,
    get_distributed_env,
    get_rank,
    get_distributed_mean,
    check_ddp_wrapped,
    check_torch_distributed_initialized,
    check_slurm_available,
    check_apex_available,
    check_amp_available,
    initialize_apex,
    assert_fp16_available,
    is_wrapped_with_ddp,
    is_torch_distributed_initialized,
    is_slurm_available,
    is_apex_available,
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
    format_metric,
    get_fn_default_params,
    get_fn_argsnames,
    get_utcnow_time,
    is_exception,
    maybe_recursive_call,
    fn_ends_with_pass,
)
from catalyst.utils.numpy import get_one_hot
from catalyst.utils.parser import parse_config_args, parse_args_uargs
from catalyst.utils.scripts import (
    import_module,
    dump_code,
    dump_python_files,
    import_experiment_and_runner,
    dump_base_experiment_code,
    distributed_cmd_run,
)
from catalyst.utils.seed import set_global_seed
from catalyst.utils.sys import (
    get_environment_vars,
    list_conda_packages,
    list_pip_packages,
    dump_environment,
)
from catalyst.utils.torch import (
    any2device,
    get_activation_fn,
    get_available_gpus,
    get_device,
    get_optimizable_params,
    get_optimizer_momentum,
    prepare_cudnn,
    process_model_params,
    set_optimizer_momentum,
    get_requires_grad,
    set_requires_grad,
    get_network_output,
    detach,
    trim_tensors,
)
from catalyst.tools.settings import IS_PRUNING_AVAILABLE

if IS_PRUNING_AVAILABLE:
    from catalyst.utils.pruning import prune_model, remove_reparametrization

from catalyst.tools.settings import IS_GIT_AVAILABLE

if IS_GIT_AVAILABLE:
    from catalyst.utils.pipelines import clone_pipeline
