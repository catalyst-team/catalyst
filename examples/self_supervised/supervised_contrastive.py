# flake8: noqa
import argparse

from common import add_arguments, ContrastiveModel, datasets, get_contrastive_model, get_loaders

import torch
import torch.nn.functional as F
from torch.optim import Adam
import torchvision

from catalyst import dl
from catalyst.contrib import nn
from catalyst.contrib.losses.supervised_contrastive import SupervisedContrastiveLoss
from catalyst.contrib.models.cv.encoders import ResnetEncoder
from catalyst.data import SelfSupervisedDatasetWrapper

parser = argparse.ArgumentParser(description="Train Supervised Contrastive")
add_arguments(parser)


def concat(*tensors):
    return torch.cat(tensors)


if __name__ == "__main__":
    args = parser.parse_args()
    batch_size = args.batch_size

    # 2. model and optimizer
    model = get_contrastive_model(args.feature_dim)
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
        loaders=get_loaders(args.dataset, args.batch_size, args.num_workers),
        verbose=True,
        logdir=args.logdir,
        valid_loader="train",
        valid_metric="loss",
        minimize_valid_metric=True,
        num_epochs=args.epochs,
    )
