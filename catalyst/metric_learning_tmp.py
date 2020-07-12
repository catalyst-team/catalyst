import catalyst.data.sampler_inbatch as si
import torchvision.transforms as t
from catalyst.contrib.datasets.metric_learning import MnistML
from catalyst.contrib.nn.criterion.triplet import TripletMarginLossWithSampling
from catalyst.contrib.nn.modules.common import Normalize
from catalyst.data.sampler import BalanceBatchSampler
from catalyst.dl.runner import SupervisedRunner
from torch import nn
from torch.nn.functional import relu, avg_pool2d, max_pool2d, log_softmax
from torch.optim import Adam
from torch.utils.data import DataLoader
import torch


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 16)
        self.norm = Normalize()

    def forward(self, x):
        x = relu(self.conv1(x))
        x = relu(self.conv2(x))
        x = max_pool2d(x, 2)
        x = torch.flatten(x, 1)
        x = relu(self.fc1(x))
        x = self.fc2(x)
        x = self.norm(x)
        return x


def metric_learning_minimal_example() -> None:
    """
    todo: add final version of it to Readme
    Draft of Metric Learning training metric_learning_minimal_example.
    By and large, to adapt this code for your task,
    all you have to do is prepare your dataset in
    the required format and select hyperparameters.
    """
    # data
    dataset = MnistML(
        root="/Users/alexeyshab/Downloads/",
        train=True,
        download=True,
        transform=t.Compose([
            t.ToTensor(),
            t.Normalize((0.1307,), (0.3081,))
        ]),
    )
    sampler = BalanceBatchSampler(labels=dataset.get_labels(), p=10, k=10)
    train_loader = DataLoader(
        dataset=dataset, sampler=sampler, batch_size=sampler.batch_size
    )

    # model
    model = Net()
    optimizer = Adam(model.parameters(), lr=1e-2)

    # criterion
    sampler_inbatch = si.HardTripletsSampler(False)
    criterion = TripletMarginLossWithSampling(
        margin=0.5, sampler_inbatch=sampler_inbatch
    )

    # train
    runner = SupervisedRunner(device="cuda:0")
    runner.train(
        model=model,
        criterion=criterion,
        optimizer=optimizer,
        loaders={"train": train_loader},
        num_epochs=100000,
        verbose=True,
    )


metric_learning_minimal_example()
