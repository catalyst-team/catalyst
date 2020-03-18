# flake8: noqa
# isort:skip_file

import logging
import os

logger = logging.getLogger(__name__)

from .argparse import boolean_flag
from .compression import pack, pack_if_needed, unpack, unpack_if_needed
from .confusion_matrix import calculate_tp_fp_fn, \
    calculate_confusion_matrix_from_arrays,\
    calculate_confusion_matrix_from_tensors
from .dataset import (
    create_dataset, split_dataset_train_test, create_dataframe
)
from .dict import (
    append_dict, flatten_dict, merge_dicts, get_dictkey_auto_fn,
    split_dict_to_subdicts
)
from .image import (
    has_image_extension, imread, imwrite, imsave, mask_to_overlay_image,
    mimread, mimwrite_with_meta, tensor_from_rgb_image, tensor_to_ndimage
)
from .misc import (
    args_are_not_none,
    make_tuple,
    pairwise,
)
from .pandas import (
    dataframe_to_list, folds_to_list, split_dataframe_train_test,
    split_dataframe_on_folds, split_dataframe_on_stratified_folds,
    split_dataframe_on_column_folds, map_dataframe, separate_tags,
    get_dataset_labeling, split_dataframe, merge_multiple_fold_csv,
    read_multiple_dataframes, read_csv_data, balance_classes
)
from .parallel import parallel_imap, tqdm_parallel_imap, get_pool
from .pipelines import clone_pipeline
from .plotly import plot_tensorboard_log
from .serialization import deserialize, serialize

try:
    import transformers  # noqa: F401
    from .text import tokenize_text, process_bert_output
except ImportError as ex:
    if os.environ.get("USE_TRANSFORMERS", "0") == "1":
        logger.warning(
            "transformers not available, to install transformers,"
            " run `pip install transformers`."
        )
        raise ex

from .visualization import (
    plot_confusion_matrix, render_figure_to_tensor, plot_metrics
)
