import warnings

warnings.filterwarnings("ignore", category=RuntimeWarning)


from catalyst.registry import Registry

from .callbacks import DetectionMeanAveragePrecision
from .criterion import SSDCriterion
from .custom_runner import SSDDetectionRunner
from .dataset import DetectionDataset
from .model import SingleShotDetector

Registry(SSDDetectionRunner)
Registry(SingleShotDetector)
Registry(SSDCriterion)
Registry(DetectionMeanAveragePrecision)
Registry(DetectionDataset)
