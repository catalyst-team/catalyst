from torch import nn
from torch.optim import Adam
from torch.utils.data import DataLoader
import torchvision.transforms as t

from catalyst.contrib.datasets.metric_learning import MnistML
from catalyst.contrib.nn.criterion.triplet import TripletMarginLossWithSampling
from catalyst.contrib.nn.modules.common import Normalize
from catalyst.data.sampler import BalanceBatchSampler
import catalyst.data.sampler_inbatch as si
from catalyst.dl.runner import SupervisedRunner


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
        transform=t.ToTensor(),
    )
    sampler = BalanceBatchSampler(labels=dataset.get_labels(), p=10, k=100)
    train_loader = DataLoader(
        dataset=dataset, sampler=sampler, batch_size=sampler.batch_size
    )

    # model
    model = nn.Sequential(*[nn.Flatten(), nn.Linear(28 * 28, 32), Normalize()])
    # from torchvision.models import resnet18
    # from catalyst.contrib.models.cv.encoders.resnet import adopt_resnet_for_1channel
    # model = adopt_resnet_for_1channel(resnet18())
    optimizer = Adam(model.parameters(), lr=1e-2)

    # criterion
    # you can also use si.AllTripletsSampler
    # sampler_inbatch = si.HardTripletsSampler(need_norm=True)
    sampler_inbatch = si.AllTripletsSampler(max_output_triplets=1000)
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
        num_epochs=1000,
        verbose=True,
    )


metric_learning_minimal_example()
