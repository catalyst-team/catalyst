from typing import Tuple

import numpy as np

from catalyst.metrics._metric import IMetric


class AdditiveValueMetric(IMetric):
    """This metric computes mean and std values of input data.

    Args:
        compute_on_call: if True, computes and returns metric value during metric call

    Examples:

    .. code-block:: python

        import numpy as np
        from catalyst import metrics

        values = [1, 2, 3, 4, 5]
        num_samples_list = [1, 2, 3, 4, 5]
        true_values = [1, 1.666667, 2.333333, 3, 3.666667]

        metric = metrics.AdditiveValueMetric()
        for value, num_samples, true_value in zip(values, num_samples_list, true_values):
            metric.update(value=value, num_samples=num_samples)
            mean, _ = metric.compute()
            assert np.isclose(mean, true_value)

    .. code-block:: python

        import os
        from torch import nn, optim
        from torch.nn import functional as F
        from torch.utils.data import DataLoader
        from catalyst import dl, metrics
        from catalyst.data.transforms import ToTensor
        from catalyst.contrib.datasets import MNIST

        model = nn.Sequential(nn.Flatten(), nn.Linear(28 * 28, 10))
        optimizer = optim.Adam(model.parameters(), lr=0.02)

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

        class CustomRunner(dl.Runner):
            def predict_batch(self, batch):
                # model inference step
                return self.model(batch[0].to(self.device))

            def on_loader_start(self, runner):
                super().on_loader_start(runner)
                self.meters = {
                    key: metrics.AdditiveValueMetric(compute_on_call=False)
                    for key in ["loss", "accuracy01", "accuracy03"]
                }

            def handle_batch(self, batch):
                # model train/valid step
                # unpack the batch
                x, y = batch
                # run model forward pass
                logits = self.model(x)
                # compute the loss
                loss = F.cross_entropy(logits, y)
                # compute other metrics of interest
                accuracy01, accuracy03 = metrics.accuracy(logits, y, topk=(1, 3))
                # log metrics
                self.batch_metrics.update(
                    {"loss": loss, "accuracy01": accuracy01, "accuracy03": accuracy03}
                )
                for key in ["loss", "accuracy01", "accuracy03"]:
                    self.meters[key].update(self.batch_metrics[key].item(), self.batch_size)
                # run model backward pass
                if self.is_train_loader:
                    loss.backward()
                    self.optimizer.step()
                    self.optimizer.zero_grad()

            def on_loader_end(self, runner):
                for key in ["loss", "accuracy01", "accuracy03"]:
                    self.loader_metrics[key] = self.meters[key].compute()[0]
                super().on_loader_end(runner)

        runner = CustomRunner()
        # model training
        runner.train(
            model=model,
            optimizer=optimizer,
            loaders=loaders,
            logdir="./logs",
            num_epochs=5,
            verbose=True,
            valid_loader="valid",
            valid_metric="loss",
            minimize_valid_metric=True,
        )

    .. note::
        Please follow the `minimal examples`_ sections for more use cases.

        .. _`minimal examples`: https://github.com/catalyst-team/catalyst#minimal-examples
    """

    def __init__(self, compute_on_call: bool = True):
        """Init AdditiveValueMetric"""
        super().__init__(compute_on_call=compute_on_call)
        self.n = 0
        self.value = 0.0
        self.mean = np.nan
        self.mean_old = 0.0
        self.m_s = 0.0
        self.std = np.nan
        self.num_samples = 0

    def reset(self) -> None:
        """Reset all fields"""
        self.n = 0
        self.value = 0.0
        self.mean = np.nan
        self.mean_old = 0.0
        self.m_s = 0.0
        self.std = np.nan
        self.num_samples = 0

    def update(self, value: float, num_samples: int) -> float:
        """Update mean metric value and std with new value.

        Args:
            value: value to update mean and std with
            num_samples: number of value samples that metrics should be updated with

        Returns:
            last value
        """
        self.value = value
        self.n += 1
        self.num_samples += num_samples

        if self.n == 1:
            # Force a copy in torch/numpy
            self.mean = 0.0 + value  # noqa: WPS345
            self.std = 0.0
            self.mean_old = self.mean
            self.m_s = 0.0
        else:
            self.mean = self.mean_old + (value - self.mean_old) * num_samples / float(
                self.num_samples
            )
            self.m_s += (value - self.mean_old) * (value - self.mean) * num_samples
            self.mean_old = self.mean
            self.std = np.sqrt(self.m_s / (self.num_samples - 1.0))
        return value

    def compute(self) -> Tuple[float, float]:
        """
        Returns mean and std values of all the input data

        Returns:
            tuple of mean and std values
        """
        return self.mean, self.std


__all__ = ["AdditiveValueMetric"]
