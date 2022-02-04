# flake8: noqa

from catalyst.settings import SETTINGS

from catalyst.contrib.data.collate_fn import FilteringCollateFn

from catalyst.contrib.data.dataset import (
    ListDataset,
    MergeDataset,
    NumpyDataset,
    PathsDataset,
)
from catalyst.contrib.data.dataset_ml import (
    MetricLearningTrainDataset,
    QueryGalleryDataset,
)

from catalyst.contrib.data.reader import (
    IReader,
    ScalarReader,
    LambdaReader,
    ReaderCompose,
)

from catalyst.contrib.data.sampler_inbatch import (
    IInbatchTripletSampler,
    InBatchTripletsSampler,
    AllTripletsSampler,
    HardTripletsSampler,
    HardClusterSampler,
)
from catalyst.contrib.data.sampler import BalanceBatchSampler, DynamicBalanceClassSampler

from catalyst.contrib.data.transforms import (
    image_to_tensor,
    normalize_image,
    Compose,
    ImageToTensor,
    NormalizeImage,
)

if SETTINGS.cv_required:
    from catalyst.contrib.data.dataset_cv import ImageFolderDataset
    from catalyst.contrib.data.reader_cv import ImageReader, MaskReader


# if SETTINGS.nifti_required:
#     from catalyst.contrib.data.reader_nifti import NiftiReader
