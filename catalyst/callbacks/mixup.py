from typing import List, Union

from catalyst.core.callback import Callback, CallbackOrder
from catalyst.core.runner import IRunner
from catalyst.utils.mixup import mixup_batch


class MixupCallback(Callback):
    """
    Callback to do mixup augmentation. More details about mixin can be found in the paper
    `mixup: Beyond Empirical Risk Minimization`: https://arxiv.org/abs/1710.09412 .

    Args:
        keys: batch keys to which you want to apply augmentation
        alpha: beta distribution a=b parameters. Must be >=0. The more alpha closer to zero the
            less effect of the mixup.
        mode: mode determines the method of use. Must be in ["replace", "add"]. If "replace"
            then replaces the batch with a mixed one, while the batch size is not changed
            If "add", concatenates mixed examples to the current ones, the batch size increases
            by 2 times.
        on_train_only: apply to train only. As the mixup use the proxy inputs, the targets are
            also proxy. We are not interested in them, are we? So, if ``on_train_only``
            is ``True`` use a standard output/metric for validation.

    Examples:

    .. code-block:: python

        from typing import Any, Dict
        import os

        import numpy as np
        import torch
        from torch import nn
        from torch.utils.data import DataLoader

        from catalyst import dl
        from catalyst.callbacks import MixupCallback
        from catalyst.contrib.datasets import MNIST
        from catalyst.data.transforms import ToTensor


        class SimpleNet(nn.Module):
            def __init__(self, in_channels, in_hw, out_features):
                super().__init__()
                self.encoder = nn.Sequential(nn.Conv2d(in_channels,
                                                       in_channels, 3, 1, 1), nn.Tanh())
                self.clf = nn.Linear(in_channels * in_hw * in_hw, out_features)

            def forward(self, x):
                features = self.encoder(x)
                features = features.view(features.size(0), -1)
                logits = self.clf(features)
                return logits


        class SimpleDataset(torch.utils.data.Dataset):
            def __init__(self, train: bool = False):
                self.mnist = MNIST(os.getcwd(), train=train, download=True, transform=ToTensor())

            def __len__(self) -> int:
                return len(self.mnist)

            def __getitem__(self, idx: int) -> Dict[str, Any]:
                x, y = self.mnist.__getitem__(idx)
                y_one_hot = np.zeros(10)
                y_one_hot[y] = 1
                return {"image": x,
                        "clf_targets": y,
                        "clf_targets_one_hot": torch.Tensor(y_one_hot)}


        model = SimpleNet(1, 28, 10)
        criterion = torch.nn.BCEWithLogitsLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.02)

        loaders = {
            "train": DataLoader(SimpleDataset(train=True), batch_size=32),
            "valid": DataLoader(SimpleDataset(train=False), batch_size=32),
        }


        class CustomRunner(dl.Runner):
            def handle_batch(self, batch):
                image = batch["image"]
                clf_logits = self.model(image)
                self.batch["clf_logits"] = clf_logits


        runner = CustomRunner()
        runner.train(
            loaders=loaders,
            model=model,
            criterion=criterion,
            optimizer=optimizer,
            logdir="./logdir14",
            num_epochs=2,
            verbose=True,
            valid_loader="valid",
            valid_metric="loss",
            minimize_valid_metric=True,
            callbacks={
                "mixup": MixupCallback(keys=["image", "clf_targets_one_hot"]),
                "criterion": dl.CriterionCallback(
                    metric_key="loss", input_key="clf_logits", target_key="clf_targets_one_hot"
                ),
                "optimizer": dl.OptimizerCallback(metric_key="loss"),
                "classification": dl.ControlFlowCallback(
                    dl.PrecisionRecallF1SupportCallback(
                        input_key="clf_logits", target_key="clf_targets", num_classes=10
                    ),
                    ignore_loaders="train",
                ),
            },
        )

    .. By running::
        With running this callback, many metrics (for example, accuracy) become undefined, so
        use ControlFlowCallback in order to evaluate model(see example)
    """

    def __init__(self, keys: Union[str, List[str]], alpha=0.2, mode="replace", on_train_only=True):
        """Init."""
        assert isinstance(keys, (str, list, tuple)), (
            f"keys must be str of list[str]," f" get: {type(keys)}"
        )
        assert alpha >= 0, "alpha must be>=0"
        assert mode in ("add", "replace"), f"mode must be in 'add', 'replace', get: {mode}"
        super().__init__(order=CallbackOrder.Internal)
        if isinstance(keys, str):
            keys = [keys]
        self.keys = keys
        self.on_train_only = on_train_only
        self.alpha = alpha
        self.mode = mode
        self.required = True

    def on_loader_start(self, runner: "IRunner") -> None:
        """
        Loader start hook.

        Args:
            runner: current runner
        """
        self.required = not self.on_train_only or runner.is_train_loader

    def on_batch_start(self, runner: "IRunner") -> None:
        """
        On batch start action.

        Args:
            runner: runner for the experiment.
        """
        if self.required:
            mixuped_batch = [runner.batch[key] for key in self.keys]
            mixuped_batch = mixup_batch(mixuped_batch, alpha=self.alpha, mode=self.mode)
            for key, mixuped_value in zip(self.keys, mixuped_batch):
                runner.batch[key] = mixuped_value


__all__ = ["MixupCallback"]
