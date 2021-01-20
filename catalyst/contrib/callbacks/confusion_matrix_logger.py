from typing import Dict, List, TYPE_CHECKING

import torch
import torch.distributed  # noqa: WPS301

from catalyst.contrib.utils.visualization import plot_confusion_matrix, render_figure_to_tensor
from catalyst.core.callback import Callback, CallbackNode, CallbackOrder
from catalyst.tools.meters.confusionmeter import ConfusionMeter

if TYPE_CHECKING:
    from catalyst.core.runner import IRunner


class ConfusionMatrixCallback(Callback):
    """Callback to plot your confusion matrix to the Tensorboard.

    Args:
        input_key: key to use from ``runner.input``, specifies our ``y_true``
        output_key: key to use from ``runner.output``, specifies our ``y_pred``
        prefix: tensorboard plot name
        mode: Strategy to compute confusion matrix.
            Must be one of [tnt, sklearn]
        class_names: list with class names
        num_classes: number of classes
        plot_params: extra params for plt.figure rendering
        tensorboard_callback_name: name of the tensorboard logger callback
    """

    def __init__(
        self,
        input_key: str = "targets",
        output_key: str = "logits",
        prefix: str = "confusion_matrix",
        mode: str = "tnt",
        class_names: List[str] = None,
        num_classes: int = None,
        plot_params: Dict = None,
        tensorboard_callback_name: str = "_tensorboard",
        version: str = None,
    ):
        """Callback initialisation."""
        super().__init__(CallbackOrder.metric, CallbackNode.all)
        self.prefix = prefix
        self.output_key = output_key
        self.input_key = input_key
        self.tensorboard_callback_name = tensorboard_callback_name

        assert mode in ["tnt"]
        assert version in [None, "tnt"]
        self._mode = version or mode
        self._plot_params = plot_params or {}

        self.class_names = class_names
        self.num_classes = num_classes if class_names is None else len(class_names)

        assert self.num_classes is not None
        self._reset_stats()

    def _reset_stats(self):
        self.confusion_matrix = ConfusionMeter(self.num_classes)

    def _add_to_stats(self, outputs, targets):
        self.confusion_matrix.add(predicted=outputs, target=targets)

    def _compute_confusion_matrix(self):
        confusion_matrix = self.confusion_matrix.value()
        return confusion_matrix

    def _plot_confusion_matrix(self, logger, epoch, confusion_matrix, class_names=None):
        fig = plot_confusion_matrix(
            confusion_matrix,
            class_names=class_names,
            normalize=True,
            show=False,
            **self._plot_params,
        )
        fig = render_figure_to_tensor(fig)
        logger.add_image(f"{self.prefix}/epoch", fig, global_step=epoch)

    def on_loader_start(self, runner: "IRunner"):
        """Loader start hook.

        Args:
            runner: current runner
        """
        self._reset_stats()

    def on_batch_end(self, runner: "IRunner"):
        """Batch end hook.

        Args:
            runner: current runner
        """
        self._add_to_stats(
            runner.output[self.output_key].detach(), runner.input[self.input_key].detach(),
        )

    def on_loader_end(self, runner: "IRunner"):
        """Loader end hook.

        Args:
            runner: current runner
        """
        class_names = self.class_names or [str(i) for i in range(self.num_classes)]
        confusion_matrix = self._compute_confusion_matrix()

        if runner.distributed_rank >= 0:
            confusion_matrix = torch.from_numpy(confusion_matrix)
            confusion_matrix = confusion_matrix.to(runner.device)
            torch.distributed.reduce(confusion_matrix, 0)
            confusion_matrix = confusion_matrix.cpu().numpy()

        if runner.distributed_rank <= 0:
            tb_callback = runner.callbacks[self.tensorboard_callback_name]
            self._plot_confusion_matrix(
                logger=tb_callback.loggers[runner.loader_key],
                epoch=runner.global_epoch,
                confusion_matrix=confusion_matrix,
                class_names=class_names,
            )


__all__ = ["ConfusionMatrixCallback"]
