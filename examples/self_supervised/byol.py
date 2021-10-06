# flake8: noqa
import argparse

from common import add_arguments, ContrastiveModel, get_loaders

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


def set_requires_grad(model, val):
    for p in model.parameters():
        p.requires_grad = val


if __name__ == "__main__":
    args = parser.parse_args()
    batch_size = args.batch_size
    aug_strength = args.aug_strength
    
    # 2. model and optimizer

    encoder_online = nn.Sequential(ResnetEncoder(arch="resnet50", frozen=False), nn.Flatten())
    projection_head_online = nn.Sequential(
        nn.Linear(2048, 512, bias=False),
        nn.ReLU(inplace=True),
        nn.Linear(512, args.feature_dim, bias=True),
    )
    encoder_target = nn.Sequential(ResnetEncoder(arch="resnet50", frozen=False), nn.Flatten())
    projection_head_target = nn.Sequential(
        nn.Linear(2048, 512, bias=False),
        nn.ReLU(inplace=True),
        nn.Linear(512, args.feature_dim, bias=True),
    )

    model = nn.ModuleDict(
        {
            "online": ContrastiveModel(projection_head_online, encoder_online),
            "target": ContrastiveModel(projection_head_target, encoder_target),
        }
    )

    set_requires_grad(model["target"], False)

    
    optimizer = Adam(model["online"].parameters(), lr=args.learning_rate)

    # 3. criterion
    criterion = NTXentLoss(tau=args.temperature)

    callbacks = [
        dl.CriterionCallback(
            input_key="online_projection_left",
            target_key="target_projection_right",
            metric_key="loss",
        ),
        dl.ControlFlowCallback(
            dl.SoftUpdateCallaback(
                target_model_key="target", source_model_key="online", tau=0.1, scope="on_batch_end"
            ),
            loaders="train",
        ),
    ]

    runner = SelfSupervisedRunner()

    runner.train(
        model=model,
        criterion=criterion,
        optimizer=optimizer,
        callbacks=callbacks,
        loaders=get_loaders(args),
        verbose=True,
        logdir=args.logdir,
        valid_loader="train",
        valid_metric="loss",
        minimize_valid_metric=True,
        num_epochs=args.epochs,
    )
