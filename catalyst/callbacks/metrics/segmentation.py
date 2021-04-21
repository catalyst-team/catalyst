from typing import List, Optional

from catalyst.callbacks.metric import BatchMetricCallback
from catalyst.metrics._segmentation import DiceMetric, IOUMetric, TrevskyMetric


class IOUCallback(BatchMetricCallback):
    """IOU metric callback.

    Args:
        input_key: input key to use for metric calculation, specifies our `y_pred`
        target_key: output key to use for metric calculation, specifies our `y_true`
        class_dim: indicates class dimension (K) for ``outputs`` and
            ``targets`` tensors (default = 1)
        weights: class weights
        class_names: class names
        threshold: threshold for outputs binarization
        log_on_batch: boolean flag to log computed metrics every batch
        prefix: metric prefix
        suffix: metric suffix

    Examples:

    .. code-block:: python

        import os
        import torch
        from torch import nn
        from torch.utils.data import DataLoader
        from catalyst import dl
        from catalyst.data.transforms import ToTensor
        from catalyst.contrib.datasets import MNIST
        from catalyst.contrib.nn import IoULoss


        model = nn.Sequential(
            nn.Conv2d(1, 1, 3, 1, 1), nn.ReLU(),
            nn.Conv2d(1, 1, 3, 1, 1), nn.Sigmoid(),
        )
        criterion = IoULoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.02)

        loaders = {
            "train": DataLoader(
                MNIST(os.getcwd(), train=True, download=True, transform=ToTensor()),
                batch_size=32
            ),
            "valid": DataLoader(
                MNIST(os.getcwd(), train=False, download=True, transform=ToTensor()),
                batch_size=32
            ),
        }

        class CustomRunner(dl.SupervisedRunner):
            def handle_batch(self, batch):
                x = batch[self._input_key]
                x_noise = (x + torch.rand_like(x)).clamp_(0, 1)
                x_ = self.model(x_noise)
                self.batch = {self._input_key: x, self._output_key: x_, self._target_key: x}

        runner = CustomRunner(
            input_key="features", output_key="scores", target_key="targets", loss_key="loss"
        )
        # model training
        runner.train(
            model=model,
            criterion=criterion,
            optimizer=optimizer,
            loaders=loaders,
            num_epochs=1,
            callbacks=[
                dl.IOUCallback(input_key="scores", target_key="targets"),
                dl.DiceCallback(input_key="scores", target_key="targets"),
                dl.TrevskyCallback(input_key="scores", target_key="targets", alpha=0.2),
            ],
            logdir="./logdir",
            valid_loader="valid",
            valid_metric="loss",
            minimize_valid_metric=True,
            verbose=True,
        )

    .. note::
        Please follow the `minimal examples`_ sections for more use cases.

        .. _`minimal examples`: https://github.com/catalyst-team/catalyst#minimal-examples
    """

    def __init__(
        self,
        input_key: str,
        target_key: str,
        class_dim: int = 1,
        weights: Optional[List[float]] = None,
        class_names: Optional[List[str]] = None,
        threshold: Optional[float] = None,
        log_on_batch: bool = True,
        prefix: str = None,
        suffix: str = None,
    ):
        """Init."""
        super().__init__(
            metric=IOUMetric(
                class_dim=class_dim,
                weights=weights,
                class_names=class_names,
                threshold=threshold,
                prefix=prefix,
                suffix=suffix,
            ),
            input_key=input_key,
            target_key=target_key,
            log_on_batch=log_on_batch,
        )


class DiceCallback(BatchMetricCallback):
    """Dice metric callback.

    Args:
        input_key: input key to use for metric calculation, specifies our `y_pred`
        target_key: output key to use for metric calculation, specifies our `y_true`
        class_dim: indicates class dimension (K) for ``outputs`` and
            ``targets`` tensors (default = 1)
        weights: class weights
        class_names: class names
        threshold: threshold for outputs binarization
        log_on_batch: boolean flag to log computed metrics every batch
        prefix: metric prefix
        suffix: metric suffix

    Examples:

    .. code-block:: python

        import os
        import torch
        from torch import nn
        from torch.utils.data import DataLoader
        from catalyst import dl
        from catalyst.data.transforms import ToTensor
        from catalyst.contrib.datasets import MNIST
        from catalyst.contrib.nn import IoULoss


        model = nn.Sequential(
            nn.Conv2d(1, 1, 3, 1, 1), nn.ReLU(),
            nn.Conv2d(1, 1, 3, 1, 1), nn.Sigmoid(),
        )
        criterion = IoULoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.02)

        loaders = {
            "train": DataLoader(
                MNIST(os.getcwd(), train=True, download=True, transform=ToTensor()),
                batch_size=32
            ),
            "valid": DataLoader(
                MNIST(os.getcwd(), train=False, download=True, transform=ToTensor()),
                batch_size=32
            ),
        }

        class CustomRunner(dl.SupervisedRunner):
            def handle_batch(self, batch):
                x = batch[self._input_key]
                x_noise = (x + torch.rand_like(x)).clamp_(0, 1)
                x_ = self.model(x_noise)
                self.batch = {self._input_key: x, self._output_key: x_, self._target_key: x}

        runner = CustomRunner(
            input_key="features", output_key="scores", target_key="targets", loss_key="loss"
        )
        # model training
        runner.train(
            model=model,
            criterion=criterion,
            optimizer=optimizer,
            loaders=loaders,
            num_epochs=1,
            callbacks=[
                dl.IOUCallback(input_key="scores", target_key="targets"),
                dl.DiceCallback(input_key="scores", target_key="targets"),
                dl.TrevskyCallback(input_key="scores", target_key="targets", alpha=0.2),
            ],
            logdir="./logdir",
            valid_loader="valid",
            valid_metric="loss",
            minimize_valid_metric=True,
            verbose=True,
        )

    .. note::
        Please follow the `minimal examples`_ sections for more use cases.

        .. _`minimal examples`: https://github.com/catalyst-team/catalyst#minimal-examples
    """

    def __init__(
        self,
        input_key: str,
        target_key: str,
        class_dim: int = 1,
        weights: Optional[List[float]] = None,
        class_names: Optional[List[str]] = None,
        threshold: Optional[float] = None,
        log_on_batch: bool = True,
        prefix: str = None,
        suffix: str = None,
    ):
        """Init."""
        super().__init__(
            metric=DiceMetric(
                class_dim=class_dim,
                weights=weights,
                class_names=class_names,
                threshold=threshold,
                prefix=prefix,
                suffix=suffix,
            ),
            input_key=input_key,
            target_key=target_key,
            log_on_batch=log_on_batch,
        )


class TrevskyCallback(BatchMetricCallback):
    """Trevsky metric callback.

    Args:
        input_key: input key to use for metric calculation, specifies our `y_pred`
        target_key: output key to use for metric calculation, specifies our `y_true`
        alpha: false negative coefficient, bigger alpha bigger penalty for
            false negative. if beta is None, alpha must be in (0, 1)
        beta: false positive coefficient, bigger alpha bigger penalty for false
            positive. Must be in (0, 1), if None beta = (1 - alpha)
        class_dim: indicates class dimension (K) for ``outputs`` and
            ``targets`` tensors (default = 1)
        weights: class weights
        class_names: class names
        threshold: threshold for outputs binarization
        log_on_batch: boolean flag to log computed metrics every batch
        prefix: metric prefix
        suffix: metric suffix

    Examples:

    .. code-block:: python

        import os
        import torch
        from torch import nn
        from torch.utils.data import DataLoader
        from catalyst import dl
        from catalyst.data.transforms import ToTensor
        from catalyst.contrib.datasets import MNIST
        from catalyst.contrib.nn import IoULoss


        model = nn.Sequential(
            nn.Conv2d(1, 1, 3, 1, 1), nn.ReLU(),
            nn.Conv2d(1, 1, 3, 1, 1), nn.Sigmoid(),
        )
        criterion = IoULoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.02)

        loaders = {
            "train": DataLoader(
                MNIST(os.getcwd(), train=True, download=True, transform=ToTensor()),
                batch_size=32
            ),
            "valid": DataLoader(
                MNIST(os.getcwd(), train=False, download=True, transform=ToTensor()),
                batch_size=32
            ),
        }

        class CustomRunner(dl.SupervisedRunner):
            def handle_batch(self, batch):
                x = batch[self._input_key]
                x_noise = (x + torch.rand_like(x)).clamp_(0, 1)
                x_ = self.model(x_noise)
                self.batch = {self._input_key: x, self._output_key: x_, self._target_key: x}

        runner = CustomRunner(
            input_key="features", output_key="scores", target_key="targets", loss_key="loss"
        )
        # model training
        runner.train(
            model=model,
            criterion=criterion,
            optimizer=optimizer,
            loaders=loaders,
            num_epochs=1,
            callbacks=[
                dl.IOUCallback(input_key="scores", target_key="targets"),
                dl.DiceCallback(input_key="scores", target_key="targets"),
                dl.TrevskyCallback(input_key="scores", target_key="targets", alpha=0.2),
            ],
            logdir="./logdir",
            valid_loader="valid",
            valid_metric="loss",
            minimize_valid_metric=True,
            verbose=True,
        )

    .. note::
        Please follow the `minimal examples`_ sections for more use cases.

        .. _`minimal examples`: https://github.com/catalyst-team/catalyst#minimal-examples
    """

    def __init__(
        self,
        input_key: str,
        target_key: str,
        alpha: float,
        beta: Optional[float] = None,
        class_dim: int = 1,
        weights: Optional[List[float]] = None,
        class_names: Optional[List[str]] = None,
        threshold: Optional[float] = None,
        log_on_batch: bool = True,
        prefix: str = None,
        suffix: str = None,
    ):
        """Init."""
        super().__init__(
            metric=TrevskyMetric(
                alpha=alpha,
                beta=beta,
                class_dim=class_dim,
                weights=weights,
                class_names=class_names,
                threshold=threshold,
                prefix=prefix,
                suffix=suffix,
            ),
            input_key=input_key,
            target_key=target_key,
            log_on_batch=log_on_batch,
        )


__all__ = ["IOUCallback", "DiceCallback", "TrevskyCallback"]
