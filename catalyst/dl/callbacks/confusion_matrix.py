from typing import Dict, List  # isort:skip

import numpy as np
from sklearn.metrics import confusion_matrix as confusion_matrix_fn

from catalyst.dl import Callback, CallbackNode, CallbackOrder, State, utils
from catalyst.utils import meters


class ConfusionMatrixCallback(Callback):
    def __init__(
        self,
        input_key: str = "targets",
        output_key: str = "logits",
        prefix: str = "confusion_matrix",
        version: str = "tnt",
        class_names: List[str] = None,
        num_classes: int = None,
        plot_params: Dict = None,
        tensorboard_callback_name: str = "_tensorboard",
    ):
        super().__init__(CallbackOrder.Metric, CallbackNode.Master)
        self.prefix = prefix
        self.output_key = output_key
        self.input_key = input_key
        self.tensorboard_callback_name = tensorboard_callback_name

        assert version in ["tnt", "sklearn"]
        self._version = version
        self._plot_params = plot_params or {}

        self.class_names = class_names
        self.num_classes = num_classes \
            if class_names is None \
            else len(class_names)

        assert self.num_classes is not None
        self._reset_stats()

    def _reset_stats(self):
        if self._version == "tnt":
            self.confusion_matrix = meters.ConfusionMeter(self.num_classes)
        elif self._version == "sklearn":
            self.outputs = []
            self.targets = []

    def _add_to_stats(self, outputs, targets):
        if self._version == "tnt":
            self.confusion_matrix.add(predicted=outputs, target=targets)
        elif self._version == "sklearn":
            outputs = outputs.cpu().numpy()
            targets = targets.cpu().numpy()

            outputs = np.argmax(outputs, axis=1)

            self.outputs.extend(outputs)
            self.targets.extend(targets)

    def _compute_confusion_matrix(self):
        if self._version == "tnt":
            confusion_matrix = self.confusion_matrix.value()
        elif self._version == "sklearn":
            confusion_matrix = confusion_matrix_fn(
                y_true=self.targets, y_pred=self.outputs
            )
        else:
            raise NotImplementedError()
        return confusion_matrix

    def _plot_confusion_matrix(
        self, logger, epoch, confusion_matrix, class_names=None
    ):
        fig = utils.plot_confusion_matrix(
            confusion_matrix,
            class_names=class_names,
            normalize=True,
            show=False,
            **self._plot_params
        )
        fig = utils.render_figure_to_tensor(fig)
        logger.add_image(f"{self.prefix}/epoch", fig, global_step=epoch)

    def on_loader_start(self, state: State):
        self._reset_stats()

    def on_batch_end(self, state: State):
        self._add_to_stats(
            state.batch_out[self.output_key].detach(),
            state.batch_in[self.input_key].detach()
        )

    def on_loader_end(self, state: State):
        class_names = \
            self.class_names or \
            [str(i) for i in range(self.num_classes)]
        confusion_matrix = self._compute_confusion_matrix()
        tb_callback = state.callbacks[self.tensorboard_callback_name]
        self._plot_confusion_matrix(
            logger=tb_callback.loggers[state.loader_name],
            epoch=state.global_epoch,
            confusion_matrix=confusion_matrix,
            class_names=class_names,
        )


__all__ = ["ConfusionMatrixCallback"]
