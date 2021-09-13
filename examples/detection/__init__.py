import warnings

warnings.filterwarnings("ignore", category=RuntimeWarning)


from catalyst.registry import Registry  # noqa: E402

from .callbacks import DetectionMeanAveragePrecision  # noqa: E402
from .criterion import CenterNetCriterion, SSDCriterion  # noqa: E402
from .custom_runner import CenterNetDetectionRunner, SSDDetectionRunner  # noqa: E402
from .dataset import CenterNetDataset, SSDDataset  # noqa: E402
from .model import CenterNet, SingleShotDetector  # noqa: E402

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
