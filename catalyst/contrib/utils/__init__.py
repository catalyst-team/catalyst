# flake8: noqa

import logging

logger = logging.getLogger(__name__)

from catalyst.settings import SETTINGS

from catalyst.contrib.utils.compression import (
    pack,
    pack_if_needed,
    unpack,
    unpack_if_needed,
)
from catalyst.contrib.utils.torch_extra import (
    calculate_tp_fp_fn,
    calculate_confusion_matrix_from_arrays,
    calculate_confusion_matrix_from_tensors,
)
from catalyst.contrib.utils.misc_extra import (
    args_are_not_none,
    make_tuple,
    pairwise,
)
from catalyst.contrib.utils.parallel import (
    parallel_imap,
    tqdm_parallel_imap,
    get_pool,
)
from catalyst.contrib.utils.serialization import deserialize, serialize
from catalyst.contrib.utils.visualization import (
    plot_confusion_matrix,
    render_figure_to_tensor,
)

try:
    import sklearn  # noqa: F401 F811
    import pandas  # noqa: F401 F811

    from catalyst.contrib.utils.pandas import (
        dataframe_to_list,
        folds_to_list,
        split_dataframe_train_test,
        split_dataframe_on_folds,
        split_dataframe_on_stratified_folds,
        split_dataframe_on_column_folds,
        map_dataframe,
        separate_tags,
        get_dataset_labeling,
        split_dataframe,
        merge_multiple_fold_csv,
        read_multiple_dataframes,
        read_csv_data,
        balance_classes,
        create_dataset,
        split_dataset_train_test,
        create_dataframe,
    )
except ModuleNotFoundError as ex:
    if SETTINGS.sklearn_required or SETTINGS.pandas_required:
        logger.warning(
            "pandas/sklearn are not available, to install them,"
            " run `pip install pandas sklearn`."
        )
        raise ex
except ImportError as ex:
    if SETTINGS.sklearn_required or SETTINGS.pandas_required:
        logger.warning(
            "pandas/sklearn are not available, to install them,"
            " run `pip install pandas sklearn`."
        )
        raise ex

try:
    import plotly  # noqa: F401
    from catalyst.contrib.utils.plotly import (
        plot_tensorboard_log,
        plot_metrics,
    )
except ModuleNotFoundError as ex:
    if SETTINGS.plotly_required:
        logger.warning(
            "plotly not available, to install plotly,"
            " run `pip install plotly`."
        )
        raise ex
except ImportError as ex:
    if SETTINGS.plotly_required:
        logger.warning(
            "plotly not available, to install plotly,"
            " run `pip install plotly`."
        )
        raise ex

try:
    from git import Repo as repo  # noqa: N813 F401
    from prompt_toolkit import prompt  # noqa: F401

    from catalyst.contrib.utils.wizard import (
        clone_pipeline,
        run_wizard,
        Wizard,
    )
except ModuleNotFoundError as ex:
    if SETTINGS.ipython_required:
        logger.warning(
            "ipython/prompt_toolkit not available, to install them,"
            " run `pip install ipython prompt_toolkit`."
        )
        raise ex
except ImportError as ex:
    if SETTINGS.ipython_required:
        logger.warning(
            "ipython/prompt_toolkit not available, to install them,"
            " run `pip install ipython prompt_toolkit`."
        )
        raise ex

from catalyst.contrib.utils.cv import *
from catalyst.contrib.utils.nlp import *
