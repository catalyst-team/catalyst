# flake8: noqa
import argparse

import torch
import torch.nn.functional as F
from torch.optim import Adam
import torchvision

from catalyst import dl
from catalyst.contrib import nn
from catalyst.contrib.models.cv.encoders import ResnetEncoder
from catalyst.contrib.nn.criterion import NTXentLoss
from catalyst.data.dataset.contrastive import ContrastiveDataset
from catalyst.runners.contrastive import ContrastiveRunner

parser = argparse.ArgumentParser(description="Train SimCLR on cifar-10")
parser.add_argument("--feature_dim", default=128, type=int, help="Feature dim for latent vector")
parser.add_argument("--temperature", default=0.5, type=float, help="Temperature used in softmax")
parser.add_argument(
    "--batch-size", default=1024, type=int, help="Number of images in each mini-batch"
)
parser.add_argument(
    "--learning-rate", default=0.001, type=float, help="Learning rate for optimizer"
)
parser.add_argument(
    "--epochs", default=100, type=int, help="Number of sweeps over the dataset to train"
)
parser.add_argument(
    "--num-workers", default=8, type=float, help="Number of workers to process a dataloader"
)
parser.add_argument(
    "--logdir", default="./logdir", type=str, help="Logs directory (tensorboard, weights, etc)",
)

parser.add_argument(
    "--aug-strength", default=1.0, type=float, help="Strength of augmentations",
)

if __name__ == "__main__":
    args = parser.parse_args()
    batch_size = args.batch_size
    aug_strength = args.aug_strength
    transforms = torchvision.transforms.Compose(
        [
            torchvision.transforms.RandomResizedCrop(32),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010]),
            torchvision.transforms.ColorJitter(
                aug_strength * 0.8, aug_strength * 0.8, aug_strength * 0.8, aug_strength * 0.2
            ),
        ]
    )

    # Cifar10MLDataset has mistakes
    # cifar_train = Cifar10MLDataset(root="./data", download=True, transform=None)

    from torchvision.datasets import CIFAR10

    cifar_train = CIFAR10(root="./data", download=True, transform=None)
    simCLR_train = ContrastiveDataset(cifar_train, transforms=transforms)
    train_loader = torch.utils.data.DataLoader(
        simCLR_train, batch_size=batch_size, num_workers=args.num_workers
    )

    # cifar_test = CifarQGDataset(root="./data", download=True)
    # valid_loader = torch.utils.data.DataLoader(
    #     simCLRDatasetWrapper(cifar_test, transforms=transforms), batch_size=batch_size, num_workers=5
    # )

    encoder = nn.Sequential(ResnetEncoder(arch="resnet50", frozen=False), nn.Flatten())
    model = nn.Sequential(
        nn.Linear(2048, 512, bias=False),
        nn.ReLU(inplace=True),
        nn.Linear(512, args.feature_dim, bias=True),
    )

    # 2. model and optimizer
    optimizer = Adam(
        [{"params": encoder.parameters()}, {"params": model.parameters()}], lr=args.learning_rate
    )

    # 3. criterion with triplets sampling
    criterion = NTXentLoss(tau=args.temperature)

    callbacks = [
        dl.ControlFlowCallback(
            dl.CriterionCallback(input_key="proj1", target_key="proj2", metric_key="loss"),
            loaders="train",
        )
    ]

    runner = ContrastiveRunner()

    runner.train(
        model=model,
        encoder=encoder,
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
