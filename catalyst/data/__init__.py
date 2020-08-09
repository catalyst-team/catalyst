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
    from catalyst.data.cv import *
except ImportError as ex:
    if settings.cv_required:
        logger.warning(
            "some of catalyst-cv dependencies not available,"
            " to install dependencies, run `pip install catalyst[cv]`."
        )
        raise ex


try:
    from catalyst.contrib.data.nlp import *
except ImportError as ex:
    if settings.nlp_required:
        logger.warning(
            "some of catalyst-nlp dependencies not available,"
            " to install dependencies, run `pip install catalyst[nlp]`."
        )
        raise ex
