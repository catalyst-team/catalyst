from typing import List, Dict
import numpy as np
from sklearn.metrics import confusion_matrix as confusion_matrix_fn
from torchnet.meter import ConfusionMeter
from tensorboardX import SummaryWriter

from catalyst.dl.core import Callback, RunnerState
from catalyst.dl import utils
from .logging import TensorboardLogger


class EarlyStoppingCallback(Callback):
    def __init__(
        self,
        patience: int,
        metric: str = "loss",
        minimize: bool = True,
        min_delta: float = 1e-6
    ):
        self.best_score = None
        self.metric = metric
        self.patience = patience
        self.num_bad_epochs = 0
        self.is_better = None

        if minimize:
            self.is_better = lambda score, best: score <= (best - min_delta)
        else:
            self.is_better = lambda score, best: score >= (best - min_delta)

    def on_epoch_end(self, state: RunnerState) -> None:
        if state.stage.startswith("infer"):
            return

        score = state.metrics.valid_values[self.metric]
        if self.best_score is None:
            self.best_score = score
        if self.is_better(score, self.best_score):
            self.num_bad_epochs = 0
            self.best_score = score
        else:
            self.num_bad_epochs += 1

        if self.num_bad_epochs >= self.patience:
            print(f"Early stop at {state.stage_epoch} epoch")
            state.early_stop = True


class ConfusionMatrixCallback(Callback):
    def __init__(
        self,
        input_key: str = "targets",
        output_key: str = "logits",
        prefix: str = "confusion_matrix",
        version: str = "tnt",
        class_names: List[str] = None,
        num_classes: int = None,
        plot_params: Dict = None
    ):
        self.prefix = prefix
        self.output_key = output_key
        self.input_key = input_key

        assert version in ["tnt", "sklearn"]
        self._version = version
        self._plot_params = plot_params or {}

        self.class_names = class_names
        self.num_classes = num_classes \
            if class_names is None \
            else len(class_names)

        assert self.num_classes is not None
        self._reset_stats()

    @staticmethod
    def _get_tensorboard_logger(state: RunnerState) -> SummaryWriter:
        # @TODO: remove this hack, simplify state
        for logger in state.loggers:
            if isinstance(logger, TensorboardLogger):
                return logger.loggers[state.loader_name]
        raise RuntimeError(
            f"Cannot find Tensorboard logger for loader {state.loader_name}")

    def _reset_stats(self):
        if self._version == "tnt":
            self.confusion_matrix = ConfusionMeter(self.num_classes)
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
                y_true=self.targets,
                y_pred=self.outputs
            )
        else:
            raise NotImplementedError()
        return confusion_matrix

    def _plot_confusion_matrix(
        self,
        logger,
        epoch,
        confusion_matrix,
        class_names=None
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

    def on_loader_start(self, state: RunnerState):
        self._reset_stats()

    def on_batch_end(self, state: RunnerState):
        self._add_to_stats(
            state.output[self.output_key].detach(),
            state.input[self.input_key].detach()
        )

    def on_loader_end(self, state: RunnerState):
        class_names = \
            self.class_names or \
            [str(i) for i in range(self.num_classes)]
        confusion_matrix = self._compute_confusion_matrix()
        logger = self._get_tensorboard_logger(state)
        self._plot_confusion_matrix(
            logger=logger,
            epoch=state.epoch,
            confusion_matrix=confusion_matrix,
            class_names=class_names
        )


__all__ = ["EarlyStoppingCallback", "ConfusionMatrixCallback"]
