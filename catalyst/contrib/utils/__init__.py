# flake8: noqa

import logging
import os

logger = logging.getLogger(__name__)

from catalyst.tools import settings

from catalyst.contrib.utils.argparse import boolean_flag
from catalyst.contrib.utils.compression import (
    pack,
    pack_if_needed,
    unpack,
    unpack_if_needed,
)
from catalyst.contrib.utils.confusion_matrix import (
    calculate_tp_fp_fn,
    calculate_confusion_matrix_from_arrays,
    calculate_confusion_matrix_from_tensors,
)
from catalyst.contrib.utils.dataset import (
    create_dataset,
    split_dataset_train_test,
    create_dataframe,
)
from catalyst.contrib.utils.misc import (
    args_are_not_none,
    make_tuple,
    pairwise,
)
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


from catalyst.contrib.utils.cv import *
from catalyst.contrib.utils.nlp import *

try:
    import plotly  # noqa: F401
    from catalyst.contrib.utils.plotly import (
        plot_tensorboard_log,
        plot_metrics,
    )
except ImportError as ex:
    if settings.plotly_required:
        logger.warning(
            "plotly not available, to install plotly,"
            " run `pip install plotly`."
        )
        raise ex
