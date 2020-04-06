import numpy as np

from catalyst.core import Callback, CallbackOrder, MetricCallback, State
from catalyst.dl import utils
from catalyst.utils import metrics

from .functional import calculate_dice


class DiceCallback(MetricCallback):
    """Dice metric callback."""

    def __init__(
        self,
        input_key: str = "targets",
        output_key: str = "logits",
        prefix: str = "dice",
        eps: float = 1e-7,
        threshold: float = None,
        activation: str = "Sigmoid",
    ):
        """
        Args:
            input_key (str): input key to use for dice calculation;
                specifies our `y_true`
            output_key (str): output key to use for dice calculation;
                specifies our `y_pred`
        """
        super().__init__(
            prefix=prefix,
            metric_fn=metrics.dice,
            input_key=input_key,
            output_key=output_key,
            eps=eps,
            threshold=threshold,
            activation=activation,
        )


class MulticlassDiceMetricCallback(Callback):
    """
    Global Multi-Class Dice Metric Callback: calculates the exact
    dice score across multiple batches. This callback is good for getting
    the dice score with small batch sizes where the batchwise dice is noisier.
    """

    def __init__(
        self,
        input_key: str = "targets",
        output_key: str = "logits",
        prefix: str = "dice",
        class_names=None,
    ):
        """
        Args:
            input_key (str): input key to use for dice calculation;
                specifies our `y_true`
            output_key (str): output key to use for dice calculation;
                specifies our `y_pred`
            prefix (str): prefix for printing the metric
            class_names (dict/List): if dictionary, should be:
                {class_id: class_name, ...} where class_id is an integer
                This allows you to ignore class indices.
                if list, make sure it corresponds to the number of classes
        """
        super().__init__(CallbackOrder.Metric)
        self.input_key = input_key
        self.output_key = output_key
        self.prefix = prefix
        self.confusion_matrix = None
        self.class_names = class_names

    def _reset_stats(self):
        """Resets the confusion matrix holding the epoch-wise stats."""
        self.confusion_matrix = None

    def on_batch_end(self, state: State):
        """Records the confusion matrix at the end of each batch.

        Args:
            state (State): current state
        """
        outputs = state.batch_out[self.output_key]
        targets = state.batch_in[self.input_key]

        confusion_matrix = utils.calculate_confusion_matrix_from_tensors(
            outputs, targets
        )

        if self.confusion_matrix is None:
            self.confusion_matrix = confusion_matrix
        else:
            self.confusion_matrix += confusion_matrix

    def on_loader_end(self, state: State):
        """@TODO: Docs. Contribution is welcome.

        Args:
            state (State): current state
        """
        tp_fp_fn_dict = utils.calculate_tp_fp_fn(self.confusion_matrix)

        dice_scores: np.ndarray = calculate_dice(**tp_fp_fn_dict)

        # logging the dice scores in the state
        for i, dice in enumerate(dice_scores):
            if isinstance(self.class_names, dict) and i not in list(
                self.class_names.keys()
            ):
                continue
            postfix = (
                self.class_names[i] if self.class_names is not None else str(i)
            )

            state.loader_metrics[f"{self.prefix}_{postfix}"] = dice

        # For supporting averaging of only classes specified in `class_names`
        values_to_avg = [
            value
            for key, value in state.loader_metrics.items()
            if key.startswith(f"{self.prefix}_")
        ]
        state.loader_metrics[f"{self.prefix}_mean"] = np.mean(values_to_avg)

        self._reset_stats()


__all__ = ["DiceCallback", "MulticlassDiceMetricCallback"]
