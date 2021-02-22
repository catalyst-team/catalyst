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

try:
    import imageio  # noqa: F401
    from skimage.color import label2rgb, rgb2gray  # noqa: F401

    from catalyst.contrib.utils.image import (
        has_image_extension,
        imread,
        imwrite,
        imsave,
        mimread,
    )
except ModuleNotFoundError as ex:
    if SETTINGS.cv_required:
        logger.warning(
            "catalyst[cv] requirements are not available, to install them,"
            " run `pip install catalyst[cv]`."
        )
        raise ex
except ImportError as ex:
    if SETTINGS.cv_required:
        logger.warning(
            "catalyst[cv] requirements are not available, to install them,"
            " run `pip install catalyst[cv]`."
        )
        raise ex

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
        balance_classes,
        create_dataset,
        split_dataset_train_test,
        create_dataframe,
    )
except ModuleNotFoundError as ex:
    if SETTINGS.ml_required:
        logger.warning(
            "catalyst[ml] requirements are not available, to install them,"
            " run `pip install catalyst[ml]`."
        )
        raise ex
except ImportError as ex:
    if SETTINGS.ml_required:
        logger.warning(
            "catalyst[ml] requirements are not available, to install them,"
            " run `pip install catalyst[ml]`."
        )
        raise ex


from catalyst.contrib.utils.parallel import (
    parallel_imap,
    tqdm_parallel_imap,
    get_pool,
)

try:
    import plotly  # noqa: F401
    from catalyst.contrib.utils.plotly import plot_metrics
except ModuleNotFoundError as ex:
    if SETTINGS.plotly_required:
        logger.warning("plotly not available, to install plotly," " run `pip install plotly`.")
        raise ex
except ImportError as ex:
    if SETTINGS.plotly_required:
        logger.warning("plotly not available, to install plotly," " run `pip install plotly`.")
        raise ex


from catalyst.contrib.utils.serialization import deserialize, serialize

try:
    import matplotlib  # noqa: F401

    from catalyst.contrib.utils.visualization import (
        plot_confusion_matrix,
        render_figure_to_tensor,
    )
except ModuleNotFoundError as ex:
    if SETTINGS.matplotlib_required:
        logger.warning(
            "matplotlib is not available, to install matplotlib," " run `pip install matplotlib`."
        )
        raise ex
except ImportError as ex:
    if SETTINGS.matplotlib_required:
        logger.warning(
            "matplotlib is not available, to install matplotlib," " run `pip install matplotlib`."
        )
        raise ex
