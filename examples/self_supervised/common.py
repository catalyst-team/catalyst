# flake8: noqa
from typing import Dict, Optional

from resnet9 import resnet9

from datasets import DATASETS
import torch
from torch.utils.data import DataLoader

from catalyst import utils
from catalyst.contrib import nn
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
        "--feature-dim", default=128, type=int, help="Feature dim for latent vector"
    )
    parser.add_argument(
        "--temperature", default=0.5, type=float, help="Temperature used in softmax"
    )
    parser.add_argument(
        "--learning-rate", default=0.001, type=float, help="Learning rate for optimizer"
    )
    utils.boolean_flag(
        parser=parser,
        name="check",
        default=False,
        help=(
            "If this flag is on the method will run only on few batches"
            "(quick test that everything working)."
        ),
    )
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

    if dataset == "STL10":
        train_dataset_kwargs = {
            "root": "data",
            "split": "train",
            "transform": None,
            "download": True,
        }
        test_dataset_kwargs = {
            "root": "data",
            "split": "test",
            "transform": None,
            "download": True,
        }
    elif dataset == "CIFAR-10":
        train_dataset_kwargs = {"root": "data", "train": True, "transform": None, "download": True}
        test_dataset_kwargs = {"root": "data", "train": False, "transform": None, "download": True}
    elif dataset == "CIFAR-100":
        train_dataset_kwargs = {"root": "data", "train": True, "transform": None, "download": True}
        test_dataset_kwargs = {"root": "data", "train": False, "transform": None, "download": True}

    train_data = SelfSupervisedDatasetWrapper(
        DATASETS[dataset]["dataset"](**train_dataset_kwargs),
        transforms=transforms,
        transform_original=transform_original,
    )
    valid_data = SelfSupervisedDatasetWrapper(
        DATASETS[dataset]["dataset"](**test_dataset_kwargs),
        transforms=transforms,
        transform_original=transform_original,
    )

    train_loader = DataLoader(train_data, batch_size=batch_size, num_workers=num_workers)
    valid_loader = DataLoader(valid_data, batch_size=batch_size, num_workers=num_workers)

    return {"train": train_loader, "valid": valid_loader}


def get_contrastive_model(
    in_size: int, feature_dim: int, encoder_dim: int = 512, hidden_dim: int = 512
) -> ContrastiveModel:
    """Init contrastive model based on parsed parametrs.

    Args:
        in_size: size of an image (in_size x in_size)
        feature_dim: dimensinality of contrative projection
        encoder_dim: dimensinality of encoder output
        hidden_dim: dimensinality of encoder-contrative projection

    Returns:
        ContrstiveModel instance
    """
    encoder = resnet9(in_size=in_size, in_channels=3, out_features=encoder_dim)
    projection_head = nn.Sequential(
        nn.Linear(encoder_dim, hidden_dim, bias=False),
        nn.ReLU(inplace=True),
        nn.Linear(hidden_dim, feature_dim, bias=True),
    )
    model = ContrastiveModel(projection_head, encoder)
    return model
