# flake8: noqa
# isort:skip_file
"""
All utils are gathered in :py:mod:`catalyst.utils` for easier access.

.. note::
    Everything from :py:mod:`catalyst.contrib.utils` is included in :py:mod:`catalyst.utils`
"""

from catalyst.contrib.utils import *

from .checkpoint import (
    load_checkpoint,
    pack_checkpoint,
    save_checkpoint,
    unpack_checkpoint,
)
from .config import load_config, save_config
from .dict import (
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
from .distributed import (
    get_nn_from_ddp_module,
    get_slurm_params,
    get_distributed_params,
    get_distributed_env,
    get_rank,
    get_distributed_mean,
    is_wrapped_with_ddp,
    is_torch_distributed_initialized,
    is_slurm_available,
    is_apex_available,
    initialize_apex,
    assert_fp16_available,
    process_components,
)
from .hash import get_hash, get_short_hash
from .initialization import get_optimal_inner_init, outer_init
from .misc import (
    copy_directory,
    format_metric,
    get_fn_default_params,
    get_fn_argsnames,
    get_utcnow_time,
    is_exception,
    maybe_recursive_call,
    fn_ends_with_pass,
)
from .numpy import get_one_hot
from .parser import parse_config_args, parse_args_uargs
from .pipelines import clone_pipeline
from .scripts import (
    import_module,
    dump_code,
    dump_python_files,
    import_experiment_and_runner,
    dump_base_experiment_code,
    distributed_cmd_run,
)
from .seed import set_global_seed
from .sys import (
    get_environment_vars,
    list_conda_packages,
    list_pip_packages,
    dump_environment,
)
from .torch import (
    any2device,
    get_activation_fn,
    get_available_gpus,
    get_device,
    get_optimizable_params,
    get_optimizer_momentum,
    prepare_cudnn,
    process_model_params,
    set_optimizer_momentum,
    set_requires_grad,
    get_network_output,
)
