import os

from kornia import augmentation
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader

from catalyst import dl
from catalyst.contrib.datasets import MNIST
from catalyst.contrib.dl.callbacks.kornia_transform import (
    BatchTransformCallback,
)
from catalyst.data.cv import ToTensor
from catalyst.utils import metrics


class CustomRunner(dl.Runner):
    """Simple runner, used to test Jupyter API features."""

    def predict_batch(self, batch):
        """Model inference step.

        Args:
            batch: batch of data

        Returns:
            batch of predictions
        """
        return self.model(batch[0].to(self.device).view(batch[0].size(0), -1))

    def _handle_batch(self, batch):
        """Model train/valid step."""
        x, y = batch
        y_hat = self.model(x.view(x.size(0), -1))

        loss = F.cross_entropy(y_hat, y)
        accuracy01, accuracy03 = metrics.accuracy(y_hat, y, topk=(1, 3))
        self.batch_metrics.update(
            {"loss": loss, "accuracy01": accuracy01, "accuracy03": accuracy03}
        )

        if self.is_train_loader:
            loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()


def main():
    """Run few epochs to check ``BatchTransformCallback`` callback."""
    model = torch.nn.Linear(28 * 28, 10)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.02)

    loaders = {
        "train": DataLoader(
            MNIST(
                os.getcwd(), train=True, download=True, transform=ToTensor()
            ),
            batch_size=32,
        ),
        "valid": DataLoader(
            MNIST(
                os.getcwd(), train=False, download=True, transform=ToTensor()
            ),
            batch_size=32,
        ),
    }

    transrorms = [
        augmentation.RandomAffine(degrees=(-15, 20), scale=(0.75, 1.25)),
    ]

    runner = CustomRunner()

    # model training
    runner.train(
        model=model,
        optimizer=optimizer,
        loaders=loaders,
        logdir="./logs",
        num_epochs=5,
        verbose=True,
        load_best_on_end=True,
        check=True,
        callbacks=[BatchTransformCallback(transrorms, input_key=0)],
    )

    # model inference
    for prediction in runner.predict_loader(loader=loaders["train"]):
        assert prediction.detach().cpu().numpy().shape[-1] == 10


if __name__ == "__main__":
    if os.getenv("USE_APEX", "0") == "0" and os.getenv("USE_DDP", "0") == "0":
        main()
