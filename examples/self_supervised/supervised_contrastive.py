# flake8: noqa
import argparse

from common import add_arguments, datasets, ContrastiveModel

import torch
import torch.nn.functional as F
from torch.optim import Adam
import torchvision

from catalyst import dl
from catalyst.contrib import nn
from catalyst.contrib.models.cv.encoders import ResnetEncoder
from catalyst.contrib.nn.criterion.supervised_contrastive import SupervisedContrastiveLoss
from catalyst.data import SelfSupervisedDatasetWrapper

parser = argparse.ArgumentParser(description="Train Supervised Contrastive on cifar-10")
add_arguments(parser)

parser.add_argument("--aug-strength", default=1.0, type=float, help="Strength of augmentations")


def concat(*tensors):
    return torch.cat(tensors)


if __name__ == "__main__":
    args = parser.parse_args()
    batch_size = args.batch_size
    aug_strength = args.aug_strength

    transforms = datasets[args.dataset]["train_transform"]
    transform_original = datasets[args.dataset]["valid_transform"]

    train_data = SelfSupervisedDatasetWrapper(
        datasets[args.dataset]["dataset"](root="data", train=True, transform=None, download=True),
        transforms=transforms,
        transform_original=transform_original,
    )
    train_loader = torch.utils.data.DataLoader(
        train_data, batch_size=batch_size, num_workers=args.num_workers
    )

    encoder = nn.Sequential(ResnetEncoder(arch="resnet50", frozen=False), nn.Flatten())
    projection_head = nn.Sequential(
        nn.Linear(2048, 512, bias=False),
        nn.ReLU(inplace=True),
        nn.Linear(512, args.feature_dim, bias=True),
    )

    model = ContrastiveModel(projection_head, encoder)
    # 2. model and optimizer
    optimizer = Adam(model.parameters(), lr=args.learning_rate)

    # 3. criterion with triplets sampling
    criterion = SupervisedContrastiveLoss(tau=args.temperature)

    callbacks = [
        dl.BatchTransformCallback(
            input_key=["projection_left", "projection_right"],
            output_key="full_projection",
            scope="on_batch_end",
            transform=concat,
        ),
        dl.BatchTransformCallback(
            input_key=["target", "target"],
            output_key="full_targets",
            scope="on_batch_end",
            transform=concat,
        ),
        dl.CriterionCallback(
            input_key="full_projection", target_key="full_targets", metric_key="loss"
        ),
    ]

    runner = dl.SelfSupervisedRunner()

    runner.train(
        model=model,
        criterion=criterion,
        optimizer=optimizer,
        callbacks=callbacks,
        loaders={
            "train": train_loader,
            # "valid": valid_loader
        },
        verbose=True,
        logdir=args.logdir,
        valid_loader="train",
        valid_metric="loss",
        minimize_valid_metric=True,
        num_epochs=args.epochs,
    )
