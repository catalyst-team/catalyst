# flake8: noqa
import argparse

from common import add_arguments, datasets

import torch
from torch.optim import Adam

from catalyst import dl
from catalyst.contrib import nn
from catalyst.contrib.models.cv.encoders import ResnetEncoder
from catalyst.contrib.nn.criterion import NTXentLoss
from catalyst.data.dataset.self_supervised import SelfSupervisedDatasetWrapper
from catalyst.dl import SelfSupervisedRunner

parser = argparse.ArgumentParser(description="Train SimCLR on cifar-10")
add_arguments(parser)

parser.add_argument("--aug-strength", default=1.0, type=float, help="Strength of augmentations")

class ContrastiveModel(torch.nn.Module):
        def __init__(self, model, encoder):
            super(ContrastiveModel, self).__init__()
            self.model = model
            self.encoder = encoder

        def forward(self, x):
            emb = self.encoder(x)
            projection = self.model(emb)
            return emb, projection

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

    encoder_online = nn.Sequential(ResnetEncoder(arch="resnet50", frozen=False), nn.Flatten())
    projection_head_online = nn.Sequential(
        nn.Linear(2048, 512, bias=False),
        nn.ReLU(inplace=True),
        nn.Linear(512, args.feature_dim, bias=True),
    )
    model_online = ContrastiveModel(projection_head_online, encoder_online)

    encoder_target = nn.Sequential(ResnetEncoder(arch="resnet50", frozen=False), nn.Flatten())
    projection_head_target = nn.Sequential(
        nn.Linear(2048, 512, bias=False),
        nn.ReLU(inplace=True),
        nn.Linear(512, args.feature_dim, bias=True),
    )
    model_target = ContrastiveModel(projection_head_target, projection_head_target)

    model = {
        "online": model_online,
        "target": model_target
    }

    # 2. model and optimizer
    optimizer = Adam(model_online.parameters(), lr=args.learning_rate)

    # 3. criterion with triplets sampling
    criterion = NTXentLoss(tau=args.temperature)

    callbacks = [
        dl.ControlFlowCallback(
            dl.CriterionCallback(
                input_key="projection_left", target_key="projection_right", metric_key="loss"
            ),
            loaders="train",
        )
    ]

    runner = SelfSupervisedRunner(encoders = ["online", "target"])

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
