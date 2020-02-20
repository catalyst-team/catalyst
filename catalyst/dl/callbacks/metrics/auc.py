from typing import List  # isort:skip

from catalyst.dl.core import MeterMetricsCallback
from catalyst.utils import meters


class AUCCallback(MeterMetricsCallback):
    """
    Calculates the AUC  per class for each loader.
    Currently, supports binary and multi-label cases.
    """
    def __init__(
        self,
        input_key: str = "targets",
        output_key: str = "logits",
        prefix: str = "auc",
        class_names: List[str] = None,
        num_classes: int = 2,
        activation: str = "Sigmoid"
    ):
        """
        Args:
            input_key (str): input key to use for auc calculation
                specifies our ``y_true``.
            output_key (str): output key to use for auc calculation;
                specifies our ``y_pred``
            prefix (str): name to display for auc when printing
            class_names (List[str]): class names to display in the logs.
                If None, defaults to indices for each class, starting from 0.
            num_classes (int): Number of classes; must be > 1
            activation (str): An torch.nn activation applied to the outputs.
                Must be one of ['none', 'Sigmoid', 'Softmax2d']
        """
        num_classes = num_classes \
            if class_names is None \
            else len(class_names)

        meter_list = [meters.AUCMeter() for _ in range(num_classes)]

        super().__init__(
            metric_names=[prefix],
            meter_list=meter_list,
            input_key=input_key,
            output_key=output_key,
            class_names=class_names,
            num_classes=num_classes,
            activation=activation
        )


__all__ = ["AUCCallback"]
