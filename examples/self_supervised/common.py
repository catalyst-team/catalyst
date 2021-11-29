from typing import Dict, Optional

from datasets import DATASETS
import torch
from torch.utils.data import DataLoader

from catalyst import utils
from catalyst.contrib import nn, ResidualBlock
from catalyst.data import SelfSupervisedDatasetWrapper


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
        "--dataset",
        default="CIFAR-10",
        type=str,
        choices=DATASETS.keys(),
        help="Dataset: CIFAR-10, CIFAR-100 or STL10",
    )
    parser.add_argument(
        "--logdir",
        default="./logdir",
        type=str,
        help="Logs directory (tensorboard, weights, etc)",
    )
    parser.add_argument(
        "--epochs", default=1000, type=int, help="Number of sweeps over the dataset to train"
    )
    parser.add_argument(
        "--num-workers", default=1, type=float, help="Number of workers to process a dataloader"
    )
    parser.add_argument(
        "--batch-size", default=512, type=int, help="Number of images in each mini-batch"
    )
    parser.add_argument(
        "--arch",
        default="resnet50",
        type=str,
        choices=["resnet18", "resnet34", "resnet50", "resnet101", "resnet152"],
    )
    utils.boolean_flag(parser=parser, name="frozen", default=False)
    parser.add_argument(
        "--feature-dim", default=128, type=int, help="Feature dim for latent vector"
    )
    parser.add_argument(
        "--temperature", default=0.5, type=float, help="Temperature used in softmax"
    )
    parser.add_argument(
        "--learning-rate", default=0.001, type=float, help="Learning rate for optimizer"
    )
    # utils.boolean_flag(parser=parser, name="check", default=False)
    utils.boolean_flag(parser=parser, name="verbose", default=False)


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


def get_loaders(
    dataset: str, batch_size: int, num_workers: Optional[int]
) -> Dict[str, DataLoader]:
    """Init loaders based on parsed parametrs.

    Args:
        dataset: dataset for the experiment
        batch_size: batch size for loaders
        num_workers: number of workers to process loaders

    Returns:
        {"train":..., "valid":...}
    """
    transforms = DATASETS[dataset]["train_transform"]
    transform_original = DATASETS[dataset]["valid_transform"]

    try:
        train_data = DATASETS[dataset]["dataset"](root="data", train=True, download=True)
        valid_data = DATASETS[dataset]["dataset"](root="data", train=False, download=True)
    except:
        train_data = DATASETS[dataset]["dataset"](root="data", split="train", download=True)
        valid_data = DATASETS[dataset]["dataset"](root="data", split="test", download=True)

    train_data = SelfSupervisedDatasetWrapper(
        train_data,
        transforms=transforms,
        transform_original=transform_original,
    )
    valid_data = SelfSupervisedDatasetWrapper(
        valid_data,
        transforms=transforms,
        transform_original=transform_original,
    )

    train_loader = DataLoader(train_data, batch_size=batch_size, num_workers=num_workers)
    valid_loader = DataLoader(valid_data, batch_size=batch_size, num_workers=num_workers)

    return {"train": train_loader, "valid": valid_loader}


def conv_block(in_channels, out_channels, pool=False):
    layers = [
        nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
    ]
    if pool:
        layers.append(nn.MaxPool2d(2))
    return nn.Sequential(*layers)


def resnet9(in_size: int, in_channels: int, out_features: int, size: int = 16):
    sz, sz2, sz4, sz8 = size, size * 2, size * 4, size * 8
    in_size = in_size // 32
    return nn.Sequential(
        conv_block(in_channels, sz),
        conv_block(sz, sz2, pool=True),
        ResidualBlock(nn.Sequential(conv_block(sz2, sz2), conv_block(sz2, sz2))),
        conv_block(sz2, sz4, pool=True),
        conv_block(sz4, sz8, pool=True),
        ResidualBlock(nn.Sequential(conv_block(sz8, sz8), conv_block(sz8, sz8))),
        nn.Sequential(
            nn.MaxPool2d(4), nn.Flatten(), nn.Dropout(0.2), nn.Linear(sz8 * in_size, out_features)
        ),
    )


def get_contrastive_model(in_size: int, feature_dim: int) -> ContrastiveModel:
    """Init contrastive model based on parsed parametrs.

    Args:
        feature_dim: dimensinality of contrative projection

    Returns:
        ContrstiveModel instance
    """
    encoder = resnet9(in_size=in_size, in_channels=3, out_features=512)
    projection_head = nn.Sequential(
        nn.Linear(512, 512, bias=False),
        nn.ReLU(inplace=True),
        nn.Linear(512, feature_dim, bias=True),
    )
    model = ContrastiveModel(projection_head, encoder)
    return model
