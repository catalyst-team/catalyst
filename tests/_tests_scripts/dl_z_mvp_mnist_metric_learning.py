import os

import torchvision.transforms as t
from torch.optim import Adam
from torch.utils.data import DataLoader

from catalyst.contrib.datasets.metric_learning import (
    MnistMLDataset,
    MnistQGDataset,
)
from catalyst.contrib.dl.callbacks import PeriodicLoaderCallback
from catalyst.contrib.models.cv.encoders.simple_conv import SimpleConv
from catalyst.contrib.nn.criterion.triplet import TripletMarginLossWithSampling
from catalyst.core.callbacks import ControlFlowCallback
from catalyst.data.sampler import BalanceBatchSampler
from catalyst.data.sampler_inbatch import AllTripletsSampler
from catalyst.dl import CriterionCallback
from catalyst.dl.callbacks.metrics.cmc import CMCScoreCallback
from catalyst.dl.runner import SupervisedRunner


def main() -> None:
    """
    Full metric learning pipeline, including train and val.
    """
    # 1. train and valid datasets
    dataset_root = "."
    transforms = t.Compose([t.ToTensor(), t.Normalize((0.1307,), (0.3081,))])

    dataset_train = MnistMLDataset(
        root=dataset_root, train=True, download=True, transform=transforms,
    )
    sampler = BalanceBatchSampler(
        labels=dataset_train.get_labels(), p=10, k=10
    )
    train_loader = DataLoader(
        dataset=dataset_train, sampler=sampler, batch_size=sampler.batch_size
    )

    dataset_val = MnistQGDataset(
        root=dataset_root, transform=transforms, gallery_fraq=0.2
    )
    val_loader = DataLoader(dataset=dataset_val, batch_size=1024)

    # 2. model and optimizer
    model = SimpleConv(input_channels=1, features_dim=16)
    optimizer = Adam(model.parameters(), lr=1e-3)

    # 3. criterion with triplets sampling
    # you can also use HardTripletsSampler
    sampler_inbatch = AllTripletsSampler(max_output_triplets=512)
    criterion = TripletMarginLossWithSampling(
        margin=0.5, sampler_inbatch=sampler_inbatch
    )

    # 4. training with catalyst Runner
    callbacks = [
        ControlFlowCallback(CriterionCallback(), loaders="train"),
        ControlFlowCallback(CMCScoreCallback(topk_args=[1]), loaders="valid"),
        PeriodicLoaderCallback(valid=100),
    ]

    runner = SupervisedRunner(device="cuda:0")
    runner.train(
        model=model,
        criterion=criterion,
        optimizer=optimizer,
        callbacks=callbacks,
        loaders={"train": train_loader, "valid": val_loader},
        minimize_metric=False,
        verbose=True,
        valid_loader="valid",
        num_epochs=1000,
        main_metric="cmc_1",
    )


if __name__ == "__main__":
    if os.getenv("USE_APEX", "0") == "0" and os.getenv("USE_DDP", "0") == "0":
        main()
