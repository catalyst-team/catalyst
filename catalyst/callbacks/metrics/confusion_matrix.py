from typing import Dict, List, TYPE_CHECKING

from catalyst.contrib.utils.visualization import (
    plot_confusion_matrix,
    render_figure_to_tensor,
)
from catalyst.core.callback import Callback, CallbackNode, CallbackOrder
from catalyst.metrics._confusion_matrix import ConfusionMatrixMetric

if TYPE_CHECKING:
    from catalyst.core.runner import IRunner


class ConfusionMatrixCallback(Callback):
    """Callback to plot your confusion matrix to the loggers.

    Args:
        input_key: key to use from ``runner.batch``, specifies our ``y_pred``
        target_key: key to use from ``runner.batch``, specifies our ``y_true``
        prefix: tensorboard plot name
        class_names: list with class names
        num_classes: number of classes
        plot_params: extra params for plt.figure rendering
    """

    def __init__(
        self,
        input_key: str = "logits",
        target_key: str = "targets",
        prefix: str = "confusion_matrix",
        class_names: List[str] = None,
        num_classes: int = None,
        normalized: bool = False,
        plot_params: Dict = None,
    ):
        """Callback initialisation."""
        super().__init__(CallbackOrder.metric, CallbackNode.master)
        assert num_classes is not None or class_names is not None
        self.prefix = prefix
        self.input_key = input_key
        self.target_key = target_key

        self._plot_params = plot_params or {}

        self.class_names = class_names or [str(i) for i in range(num_classes)]
        self.num_classes = (
            num_classes if class_names is None else len(class_names)
        )
        self.normalized = normalized

        assert self.num_classes is not None

    def on_loader_start(self, runner: "IRunner"):
        """Loader start hook.

        Args:
            runner: current runner
        """
        self.confusion_matrix = ConfusionMatrixMetric(
            num_classes=self.num_classes, normalized=self.normalized
        )

    def on_batch_end(self, runner: "IRunner"):
        """Batch end hook.

        Args:
            runner: current runner
        """
        inputs, targets = (
            runner.batch[self.input_key].detach(),
            runner.batch[self.target_key].detach(),
        )
        inputs, targets = (
            runner.engine.sync_tensor(inputs),
            runner.engine.sync_tensor(targets),
        )
        self.confusion_matrix.update(predictions=inputs, targets=targets)

    def on_loader_end(self, runner: "IRunner"):
        """Loader end hook.

        Args:
            runner: current runner
        """
        confusion_matrix = self.confusion_matrix.compute()
        # if runner.engine.rank >= 0:
        #     confusion_matrix = torch.from_numpy(confusion_matrix)
        #     confusion_matrix = runner.engine.sync_tensor(confusion_matrix)
        #     confusion_matrix = confusion_matrix.cpu().numpy()
        # if runner.engine.rank <= 0:
        fig = plot_confusion_matrix(
            confusion_matrix,
            class_names=self.class_names,
            normalize=self.normalized,
            show=False,
            **self._plot_params,
        )
        image = render_figure_to_tensor(fig)
        runner.log_image(tag=self.prefix, image=image, scope="loader")


__all__ = ["ConfusionMatrixCallback"]
