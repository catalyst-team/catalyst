import os

import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import MNIST

from catalyst import dl
from catalyst.utils import metrics

model = torch.nn.Linear(28 * 28, 10)
optimizer = torch.optim.Adam(model.parameters(), lr=0.02)

loaders = {
    "train": DataLoader(
        MNIST(
            os.getcwd(),
            train=False,
            download=True,
            transform=transforms.ToTensor(),
        ),
        batch_size=32,
    ),
    "valid": DataLoader(
        MNIST(
            os.getcwd(),
            train=False,
            download=True,
            transform=transforms.ToTensor(),
        ),
        batch_size=32,
    ),
}


class CustomRunner(dl.Runner):
    """
    Docs.
    """

    def _handle_batch(self, batch):
        """
        Docs.
        """
        x, y = batch
        y_hat = self.model(x.view(x.size(0), -1))
        loss = F.cross_entropy(y_hat, y)
        accuracy01, accuracy03, accuracy05 = metrics.accuracy(
            y_hat, y, topk=(1, 3, 5)
        )

        self.state.batch_metrics = {
            "loss": loss,
            "accuracy01": accuracy01,
            "accuracy03": accuracy03,
            "accuracy05": accuracy05,
        }

        if self.state.is_train_loader:
            loss.backward()
            self.state.optimizer.step()
            self.state.optimizer.zero_grad()


runner = CustomRunner()
runner.train(
    model=model,
    optimizer=optimizer,
    loaders=loaders,
    verbose=True,
    check=True,
)
