import numpy as np
import torch

from catalyst.core import Callback, CallbackOrder, IRunner


class MixupCallback(Callback):
    """Callback to do mixup augmentation.

    More details about mixin can be found in the paper `mixup: Beyond Empirical Risk Minimization.

    Examples:

    .. code-block:: python

        import os
        import torch
        from torch import nn
        from torch.utils.data import DataLoader
        from typing import Dict, Any
        from catalyst import dl
        from catalyst.callbacks import MixupCallback
        from catalyst.data.transforms import ToTensor
        from catalyst.contrib.datasets import MNIST


        class SimpleNet(nn.Module):

            def __init__(self, in_channels, in_hw, out_features):
                super().__init__()
                self.encoder = nn.Sequential(nn.Conv2d(in_channels,
                                                       in_channels, 3, 1, 1), nn.Tanh())
                self.clf = nn.Linear(in_channels * in_hw * in_hw, out_features)

            def forward(self, x):
                z = self.encoder(x)
                z_ = z.view(z.size(0), -1)
                y_hat = self.clf(z_)
                return y_hat


        class SimpleDataset(torch.utils.data.Dataset):
            def __init__(self):
                self.mnist = MNIST(os.getcwd(), train=False, download=True, transform=ToTensor())

            def __len__(self) -> int:
                return len(self.mnist)

            def __getitem__(self, idx: int) -> Dict[str, Any]:
                x, y = self.mnist.__getitem__(idx)
                y_one_hot = np.zeros(10)
                y_one_hot[y] = 1
                return {'image': x, 'clf_targets_one_hot': torch.Tensor(y_one_hot)}


        model = SimpleNet(1, 28, 10)
        criterion = torch.nn.BCEWithLogitsLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.02)

        loaders = {
            "train": DataLoader(SimpleDataset(), batch_size=32),
            "valid": DataLoader(SimpleDataset(), batch_size=32),
        }


        class CustomRunner(dl.Runner):

            def handle_batch(self, batch):
                image = batch['image']
                clf_logits = self.model(image)
                self.batch['clf_logits'] = clf_logits


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
                "mixup": MixupCallback(input_key='image', output_key='clf_targets_one_hot'),
                "criterion": dl.CriterionCallback(
                    metric_key="loss",
                    input_key="clf_logits",
                    target_key="clf_targets_one_hot",
                ),
                "optimizer": dl.OptimizerCallback(metric_key="loss"),
            },
        )
        
    .. note::
        1) Callback can only be used with an even batch size
        2) With running this callback, many metrics (for example, accuracy) become undefined.
    """

    def __init__(
        self,
        input_key: str = "targets",
        output_key: str = "logits",
        alpha=0.2,
        mode="replace",
        on_train_only=True,
        **kwargs,
    ):
        """

        Args:
            input_key: feature key to which you want to apply augmentation
            output_key: target key to which you want to apply augmentation
            alpha: beta distribution a=b parameters. Must be >=0. The more alpha closer to zero the
            less effect of the mixup.
            mode: mode determines the method of use. Must be in ["replace", "add"]. If "replace"
            then replaces the batch with a mixed one, while the batch size is reduced by 2 times.
            If "add", concatenates mixed examples to the current ones, the batch size increases by
            1.5 times.
            on_train_only: apply to train only. As the mixup use the proxy inputs, the targets are
            also proxy. We are not interested in them, are we? So, if on_train_only is True, use a
            standard output/metric for validation.
            **kwargs:
        """
        assert isinstance(input_key, str) and isinstance(output_key, str)
        assert alpha >= 0, "alpha must be>=0"
        assert mode in ["add", "replace"], f"mode must be in 'add', 'replace', get: {mode}"
        super().__init__(order=CallbackOrder.Internal)
        self.input_key = input_key
        self.output_key = output_key
        self.on_train_only = on_train_only
        self.alpha = alpha
        self.mode = mode
        self.is_needed = True

    def _handle_batch(self, runner: "IRunner") -> None:
        """
        Applies mixup augmentation for a batch

        Args:
            runner: runner for the experiment.
        """
        features = runner.batch[self.input_key]
        targets = runner.batch[self.output_key]

        batch_size = features.shape[0]
        assert batch_size % 2 == 0
        beta = np.random.beta(self.alpha, self.alpha, batch_size // 2).astype(np.float32)

        # It is need in order to word with different dimensions
        features_shape = [batch_size // 2] + [1] * len(features.shape[1:])
        targets_shape = [batch_size // 2] + [1] * len(targets.shape[1:])

        features_beta = beta.reshape(features_shape)
        beta = beta.reshape(targets_shape)

        indexes = np.array(list(range(batch_size)))
        np.random.shuffle(indexes)

        features = features[indexes[: batch_size // 2]] * features_beta + features[
            indexes[batch_size // 2 :]
        ] * (1 - features_beta)

        targets = targets[indexes[: batch_size // 2]] * beta + targets[
            indexes[batch_size // 2 :]
        ] * (1 - beta)

        if self.mode == "replace":
            runner.batch[self.input_key] = features
            runner.batch[self.output_key] = targets
        else:
            # self.mode == 'add':
            runner.batch[self.input_key] = torch.cat([runner.batch[self.input_key], features])
            runner.batch[self.output_key] = torch.cat([runner.batch[self.output_key], targets])

    def on_loader_start(self, runner: "IRunner") -> None:
        """
        Loader start hook.

        Args:
            runner: current runner
        """
        self.is_needed = not self.on_train_only or runner.is_train_loader

    def on_batch_start(self, runner: "IRunner") -> None:
        """
        On batch start action.

        Args:
            runner: runner for the experiment.
        """
        if self.is_needed:
            self._handle_batch(runner)


__all__ = ["MixupCallback"]
