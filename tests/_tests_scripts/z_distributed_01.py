import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from catalyst import dl


class Projector(nn.Module):
    """
    Simpler neural network example - Projector.
    """

    def __init__(self, input_size: int):
        """
        Init method.

        Args:
            input_size(int): number of features in projected space.
        """
        super().__init__()
        self.linear = nn.Linear(input_size, 1)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """
        Forward method.

        Args:
            X(torch.Tensor): input tensor.

        Returns:
            (torch.Tensor): output projection.
        """
        return self.linear(X).squeeze(-1)


X = torch.rand(int(1e4), 10)
y = torch.rand(X.shape[0])
model = Projector(X.shape[1])
dataset = TensorDataset(X, y)
loader = DataLoader(dataset, batch_size=32, num_workers=1)


# example  1 - typical training
runner = dl.SupervisedRunner()
runner.train(
    model=model,
    loaders={"train": loader, "valid": loader},
    # datasets={
    #     "batch_size": 32,
    #     "num_workers": 1,
    #     "train": dataset,
    #     "valid": dataset,
    # },
    criterion=nn.MSELoss(),
    optimizer=optim.Adam(model.parameters()),
    logdir="logs/log_example_01",
    num_epochs=10,
    verbose=True,
    check=True,
    fp16=False,
    distributed=False,
)
