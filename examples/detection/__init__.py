import warnings

warnings.filterwarnings("ignore", category=RuntimeWarning)


from catalyst.registry import Registry  # noqa: E402

from .callbacks import DetectionMeanAveragePrecision  # noqa: E402
from .criterion import CenterNetCriterion, SSDCriterion  # noqa: E402
from .custom_runner import (  # noqa: E402
    CenterNetDetectionRunner,
    SSDDetectionRunner,
    YOLOXDetectionRunner,
)
from .dataset import CenterNetDataset, SSDDataset, YOLOXDataset  # noqa: E402
from .models import (  # noqa: E402
    CenterNet,
    SingleShotDetector,
    yolo_x_tiny,
    yolo_x_small,
    yolo_x_medium,
    yolo_x_large,
    yolo_x_big,
)

# runers
Registry(SSDDetectionRunner)
Registry(CenterNetDetectionRunner)
Registry(YOLOXDetectionRunner)

# models
Registry(SingleShotDetector)
Registry(CenterNet)
Registry(yolo_x_tiny)
Registry(yolo_x_small)
Registry(yolo_x_medium)
Registry(yolo_x_large)
Registry(yolo_x_big)

# criterions
Registry(SSDCriterion)
Registry(CenterNetCriterion)

# callbacks
Registry(DetectionMeanAveragePrecision)

# datasets
Registry(SSDDataset)
Registry(CenterNetDataset)
Registry(YOLOXDataset)
