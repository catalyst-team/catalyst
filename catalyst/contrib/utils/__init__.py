# flake8: noqa

from catalyst.settings import SETTINGS

from catalyst.contrib.utils.compression import (
    pack,
    pack_if_needed,
    unpack,
    unpack_if_needed,
)

if SETTINGS.cv_required:
    from catalyst.contrib.utils.image import (
        has_image_extension,
        imread,
        imwrite,
        imsave,
        mimread,
    )

if SETTINGS.ml_required:
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


from catalyst.contrib.utils.parallel import (
    parallel_imap,
    tqdm_parallel_imap,
    get_pool,
)


from catalyst.contrib.utils.serialization import deserialize, serialize

if SETTINGS.ml_required:
    from catalyst.contrib.utils.thresholds import (
        get_baseline_thresholds,
        get_binary_threshold,
        get_multiclass_thresholds,
        get_multilabel_thresholds,
        get_binary_threshold_cv,
        get_multilabel_thresholds_cv,
        get_thresholds_greedy,
        get_multilabel_thresholds_greedy,
        get_multiclass_thresholds_greedy,
        get_best_multilabel_thresholds,
        get_best_multiclass_thresholds,
    )

if SETTINGS.ml_required:
    from catalyst.contrib.utils.visualization import (
        plot_confusion_matrix,
        render_figure_to_array,
    )
