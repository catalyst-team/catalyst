from typing import Dict

from datasets import datasets

import torch
from torch.utils.data import DataLoader

from catalyst.contrib import nn
from catalyst.contrib.models.cv.encoders import ResnetEncoder
from catalyst.data.dataset.self_supervised import SelfSupervisedDatasetWrapper


def add_arguments(parser) -> None:
    """Function to add common arguments to argparse:
    feature_dim: Feature dim for latent vector
    temperature: Temperature used in softmax
    batch_size: Number of images in each mini-batch
    epochs: Number of sweeps over the dataset to train
    num_workers: Number of workers to process a dataloader
    logdir: Logs directory (tensorboard, weights, etc)
    dataset: CIFAR-10, CIFAR-100 or STL10
    learning-rate: Learning rate for optimizer

    Args:
        parser: argparser like object
    """
    parser.add_argument(
        "--feature_dim", default=128, type=int, help="Feature dim for latent vector"
    )
    parser.add_argument(
        "--temperature", default=0.5, type=float, help="Temperature used in softmax"
    )
    parser.add_argument(
        "--batch_size", default=512, type=int, help="Number of images in each mini-batch"
    )
    parser.add_argument(
        "--epochs", default=1000, type=int, help="Number of sweeps over the dataset to train"
    )
    parser.add_argument(
        "--num_workers", default=8, type=float, help="Number of workers to process a dataloader"
    )
    parser.add_argument(
        "--logdir",
        default="./logdir",
        type=str,
        help="Logs directory (tensorboard, weights, etc)",
    )
    parser.add_argument(
        "--dataset",
        default="CIFAR-10",
        type=str,
        choices=datasets.keys(),
        help="Dataset: CIFAR-10, CIFAR-100 or STL10",
    )
    parser.add_argument(
        "--learning-rate", default=0.001, type=float, help="Learning rate for optimizer"
    )


class ContrastiveModel(torch.nn.Module):
    """Contrastive model with projective head.

    Args:
        model: projective head for the train time
        encoder: model for the future uses
    """

    def __init__(self, model, encoder):
        super(ContrastiveModel, self).__init__()
        self.model = model
        self.encoder = encoder

    def forward(self, x):
        """Forward method.

        Args:
            x: input for the encoder

        Returns:
            (embeddings, projections)
        """
        emb = self.encoder(x)
        projection = self.model(emb)
        return emb, projection


def get_loaders(args) -> Dict[str, DataLoader]:
    """Init loaders based on parsed parametrs.

    Args:
        args: argparse parametrs

    Returns:
        {"train":..., "valid":...}
    """
    transforms = datasets[args.dataset]["train_transform"]
    transform_original = datasets[args.dataset]["valid_transform"]

    train_data = SelfSupervisedDatasetWrapper(
        datasets[args.dataset]["dataset"](root="data", train=True, transform=None, download=True),
        transforms=transforms,
        transform_original=transform_original,
    )
    valid_data = SelfSupervisedDatasetWrapper(
        datasets[args.dataset]["dataset"](root="data", train=False, transform=None, download=True),
        transforms=transforms,
        transform_original=transform_original,
    )

    train_loader = DataLoader(train_data, batch_size=args.batch_size, num_workers=args.num_workers)

    valid_loader = DataLoader(valid_data, batch_size=args.batch_size, num_workers=args.num_workers)

    return {"train": train_loader, "valid": valid_loader}


def get_contrastive_model(args) -> ContrastiveModel:
    """Init contrastive model based on parsed parametrs.

    Args:
        args: argparse parametrs

    Returns:
        ContrstiveModel instance
    """
    encoder = nn.Sequential(ResnetEncoder(arch="resnet50", frozen=False), nn.Flatten())
    projection_head = nn.Sequential(
        nn.Linear(2048, 512, bias=False),
        nn.ReLU(inplace=True),
        nn.Linear(512, args.feature_dim, bias=True),
    )
    model = ContrastiveModel(projection_head, encoder)
    return model
