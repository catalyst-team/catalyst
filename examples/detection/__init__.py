import warnings

warnings.filterwarnings("ignore", category=RuntimeWarning)


from catalyst.registry import Registry

from .callbacks import DetectionMeanAveragePrecision
from .criterion import CenterNetCriterion, SSDCriterion
from .custom_runner import CenterNetDetectionRunner, SSDDetectionRunner
from .dataset import CenterNetDataset, SSDDataset
from .model import CenterNet, SingleShotDetector

# runers
Registry(SSDDetectionRunner)
Registry(CenterNetDetectionRunner)

# models
Registry(SingleShotDetector)
Registry(CenterNet)

# criterions
Registry(SSDCriterion)
Registry(CenterNetCriterion)

# callbacks
Registry(DetectionMeanAveragePrecision)

# datasets
Registry(SSDDataset)
Registry(CenterNetDataset)
