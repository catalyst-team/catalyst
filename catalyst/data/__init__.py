# flake8: noqa
import logging

from catalyst.tools import settings

from catalyst.data.augmentor import Augmentor, AugmentorCompose, AugmentorKeys
from catalyst.data.collate_fn import FilteringCollateFn
from catalyst.data.dataset import (
    DatasetFromSampler,
    ListDataset,
    MergeDataset,
    NumpyDataset,
    PathsDataset,
)
from catalyst.data.loader import BatchLimitLoaderWrapper
from catalyst.data.reader import (
    ReaderSpec,
    ScalarReader,
    LambdaReader,
    ReaderCompose,
)
from catalyst.data.sampler import (
    BalanceClassSampler,
    BalanceBatchSampler,
    DistributedSamplerWrapper,
    DynamicLenBatchSampler,
    MiniEpochSampler,
)
from catalyst.data.sampler_inbatch import (
    InBatchTripletsSampler,
    AllTripletsSampler,
    HardTripletsSampler,
    IInbatchTripletSampler,
    HardClusterSampler,
)

logger = logging.getLogger(__name__)

try:
    # @TODO: remove hotfix
    from catalyst.contrib.data.cv.reader import (  # noqa: F401
        ImageReader,
        MaskReader,
    )
except ImportError as ex:
    if settings.cv_required:
        logger.warning(
            "some of catalyst-cv dependencies not available,"
            " to install dependencies, run `pip install catalyst[cv]`."
        )
        raise ex
