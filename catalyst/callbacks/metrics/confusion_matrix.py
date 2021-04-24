from typing import Dict, List, TYPE_CHECKING

from catalyst.core.callback import Callback, CallbackNode, CallbackOrder
from catalyst.metrics._confusion_matrix import ConfusionMatrixMetric
from catalyst.settings import SETTINGS

if SETTINGS.ml_required:
    from catalyst.contrib.utils.visualization import plot_confusion_matrix, render_figure_to_array

if TYPE_CHECKING:
    from catalyst.core.runner import IRunner


class ConfusionMatrixCallback(Callback):
    """Callback to plot your confusion matrix to the loggers.

    Args:
        input_key: key to use from ``runner.batch``, specifies our ``y_pred``
        target_key: key to use from ``runner.batch``, specifies our ``y_true``
        prefix: plot name for monitoring tools
        class_names: list with class names
        num_classes: number of classes
        normalized: boolean flag for confusion matrix normalization
        plot_params: extra params for plt.figure rendering

    .. note::
        catalyst[ml] required for this callback

    Examples:

    .. code-block:: python

        import torch
        from torch.utils.data import DataLoader, TensorDataset
        from catalyst import dl

        # sample data
        num_samples, num_features, num_classes = int(1e4), int(1e1), 4
        X = torch.rand(num_samples, num_features)
        y = (torch.rand(num_samples,) * num_classes).to(torch.int64)

        # pytorch loaders
        dataset = TensorDataset(X, y)
        loader = DataLoader(dataset, batch_size=32, num_workers=1)
        loaders = {"train": loader, "valid": loader}

        # model, criterion, optimizer, scheduler
        model = torch.nn.Linear(num_features, num_classes)
        criterion = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters())
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, [2])

        # model training
        runner = dl.SupervisedRunner(
            input_key="features", output_key="logits", target_key="targets", loss_key="loss"
        )
        runner.train(
            model=model,
            criterion=criterion,
            optimizer=optimizer,
            scheduler=scheduler,
            loaders=loaders,
            logdir="./logdir",
            num_epochs=3,
            valid_loader="valid",
            valid_metric="accuracy03",
            minimize_valid_metric=False,
            verbose=True,
            callbacks=[
                dl.AccuracyCallback(
                    input_key="logits", target_key="targets", num_classes=num_classes
                ),
                dl.PrecisionRecallF1SupportCallback(
                    input_key="logits", target_key="targets", num_classes=num_classes
                ),
                dl.AUCCallback(input_key="logits", target_key="targets"),
                dl.ConfusionMatrixCallback(
                    input_key="logits", target_key="targets", num_classes=num_classes
                ),
            ],
        )

    .. note::
        Please follow the `minimal examples`_ sections for more use cases.

        .. _`minimal examples`: https://github.com/catalyst-team/catalyst#minimal-examples
    """

    def __init__(
        self,
        input_key: str,
        target_key: str,
        prefix: str = None,
        class_names: List[str] = None,
        num_classes: int = None,
        normalized: bool = False,
        plot_params: Dict = None,
    ):
        """Callback initialisation."""
        super().__init__(CallbackOrder.metric, CallbackNode.all)
        assert num_classes is not None or class_names is not None
        self.prefix = prefix or "confusion_matrix"
        self.input_key = input_key
        self.target_key = target_key

        self._plot_params = plot_params or {}

        self.class_names = class_names or [f"class_{i:02d}" for i in range(num_classes)]
        self.num_classes = num_classes if class_names is None else len(class_names)
        self.normalized = normalized

        assert self.num_classes is not None
        self.confusion_matrix = ConfusionMatrixMetric(
            num_classes=self.num_classes, normalized=self.normalized
        )

    def on_loader_start(self, runner: "IRunner"):
        """Loader start hook.

        Args:
            runner: current runner
        """
        self.confusion_matrix.reset()

    def on_batch_end(self, runner: "IRunner"):
        """Batch end hook.

        Args:
            runner: current runner
        """
        inputs, targets = (
            runner.batch[self.input_key].detach(),
            runner.batch[self.target_key].detach(),
        )
        self.confusion_matrix.update(predictions=inputs, targets=targets)

    def on_loader_end(self, runner: "IRunner"):
        """Loader end hook.

        Args:
            runner: current runner
        """
        confusion_matrix = self.confusion_matrix.compute()
        fig = plot_confusion_matrix(
            confusion_matrix,
            class_names=self.class_names,
            normalize=self.normalized,
            show=False,
            **self._plot_params,
        )
        image = render_figure_to_array(fig)
        runner.log_image(tag=self.prefix, image=image, scope="loader")


__all__ = ["ConfusionMatrixCallback"]
