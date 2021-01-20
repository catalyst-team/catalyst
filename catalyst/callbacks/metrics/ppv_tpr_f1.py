from typing import List

from catalyst.callbacks.meter import MeterMetricsCallback
from catalyst.tools.meters.ppv_tpr_f1_meter import PrecisionRecallF1ScoreMeter


class PrecisionRecallF1ScoreCallback(MeterMetricsCallback):
    """
    Calculates the global precision (positive predictive value or ppv),
    recall (true positive rate or tpr), and F1-score per class for each loader.

    .. note::
        Currently, supports binary and multilabel cases.
    """

    def __init__(
        self,
        input_key: str = "targets",
        output_key: str = "logits",
        class_names: List[str] = None,
        num_classes: int = 2,
        threshold: float = 0.5,
        activation: str = "Sigmoid",
    ):
        """
        Args:
            input_key: input key to use for metric calculation
                specifies our ``y_true``
            output_key: output key to use for metric calculation;
                specifies our ``y_pred``
            class_names: class names to display in the logs.
                If None, defaults to indices for each class, starting from 0.
            num_classes: Number of classes; must be > 1
            threshold: threshold for outputs binarization
            activation: An torch.nn activation applied to the outputs.
                Must be one of ``'none'``, ``'Sigmoid'``, ``'Softmax'``
        """
        # adjusting num_classes automatically if class_names is not None
        num_classes = num_classes if class_names is None else len(class_names)

        meter_list = [PrecisionRecallF1ScoreMeter(threshold) for _ in range(num_classes)]

        super().__init__(
            metric_names=["ppv", "tpr", "f1"],
            meter_list=meter_list,
            input_key=input_key,
            output_key=output_key,
            class_names=class_names,
            num_classes=num_classes,
            activation=activation,
        )


__all__ = ["PrecisionRecallF1ScoreCallback"]
