# flake8: noqa
import argparse

import torch
import torch.nn.functional as F
from torch.optim import Adam
import torchvision

from catalyst import dl
from catalyst.contrib import nn
from catalyst.contrib.models.cv.encoders import ResnetEncoder
from catalyst.contrib.nn.criterion.supervised_contrastive import SupervisedContrastiveLoss
from catalyst.data import SelfSupervisedDatasetWrapper
from catalyst.dl import SelfSupervisedRunner

parser = argparse.ArgumentParser(description="Train SimCLR on cifar-10")
parser.add_argument("--feature_dim", default=128, type=int, help="Feature dim for latent vector")
parser.add_argument("--temperature", default=0.5, type=float, help="Temperature used in softmax")
parser.add_argument(
    "--batch-size", default=128, type=int, help="Number of images in each mini-batch"
)
parser.add_argument(
    "--learning-rate", default=0.001, type=float, help="Learning rate for optimizer"
)
parser.add_argument(
    "--epochs", default=100, type=int, help="Number of sweeps over the dataset to train"
)
parser.add_argument(
    "--num-workers", default=0, type=float, help="Number of workers to process a dataloader"
)
parser.add_argument(
    "--logdir", default="./logdir", type=str, help="Logs directory (tensorboard, weights, etc)",
)

parser.add_argument(
    "--aug-strength", default=1.0, type=float, help="Strength of augmentations",
)


def concat(*tensors):
    return torch.cat(tensors)


if __name__ == "__main__":
    args = parser.parse_args()
    batch_size = args.batch_size
    aug_strength = args.aug_strength
    transforms = torchvision.transforms.Compose(
        [
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010]),
            torchvision.transforms.RandomResizedCrop(32),
            torchvision.transforms.ColorJitter(
                aug_strength * 0.8, aug_strength * 0.8, aug_strength * 0.8, aug_strength * 0.2
            ),
        ]
    )

    transform_original = transforms = torchvision.transforms.Compose(
        [
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010]),
        ]
    )

    from torchvision.datasets import CIFAR10

    cifar_train = CIFAR10(root="./data", download=True, transform=None)
    simCLR_train = SelfSupervisedDatasetWrapper(
        cifar_train, transforms=transforms, transform_original=transform_original
    )
    train_loader = torch.utils.data.DataLoader(
        simCLR_train, batch_size=batch_size, num_workers=args.num_workers
    )

    encoder = nn.Sequential(ResnetEncoder(arch="resnet50", frozen=False), nn.Flatten())
    projection_head = nn.Sequential(
        nn.Linear(2048, 512, bias=False),
        nn.ReLU(inplace=True),
        nn.Linear(512, args.feature_dim, bias=True),
    )

    class ContrastiveModel(torch.nn.Module):
        def __init__(self, model, encoder):
            super(ContrastiveModel, self).__init__()
            self.model = model
            self.encoder = encoder

        def forward(self, x):
            emb = self.encoder(x)
            projection = self.model(emb)
            return emb, projection

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

    runner = SelfSupervisedRunner()

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
