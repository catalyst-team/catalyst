from typing import List, Union
from abc import ABC, abstractmethod

from torch import int as tint, long, nn, short, Tensor
from torch.nn import TripletMarginLoss
from torch.optim import Adam
from torch.utils.data import DataLoader, Dataset
from torchvision.datasets import Omniglot
from torchvision.models import ResNet, resnet18
import torchvision.transforms as t

from catalyst.data.sampler import BalanceBatchSampler
import catalyst.data.sampler_inbatch as si
from catalyst.dl.runner import SupervisedRunner


def adopt_resnet_for_1channel(model: ResNet) -> ResNet:
    """
    Args:
        model: ResNet model

    Returns:
        ResNet model with changed 1st conv layer
    """
    conv_old = model.conv1
    model.conv1 = nn.Conv2d(
        in_channels=1,
        out_channels=conv_old.out_channels,
        kernel_size=conv_old.kernel_size,
        stride=conv_old.stride,
        padding=conv_old.padding,
        bias=conv_old.bias,
    )
    return model


class MetricLearningTrainDataset(Dataset, ABC):
    """
    Base class for datasets adapted for
    metric learning train stage.
    """

    @abstractmethod
    def get_labels(self) -> List[int]:
        """
        Dataset for metric learning must provide
        label of each sample for forming positive
        and negative pairs during the training
        based on these labels.

        Returns:
            labels of samples
        """
        raise NotImplemented


class OmniglotML(MetricLearningTrainDataset, Omniglot):
    """
    Simple wrapper for Omniglot dataset
    """

    def get_labels(self) -> List[int]:
        """
        Returns:
            labels of symbols
        """
        return [label for _, label in self._flat_character_images]


class TripletMarginLossWithSampling(nn.Module):
    """
    This class combains in-batch sampling of triplets and
    default TripletMargingLoss from PyTorch.
    """

    def __init__(
        self, margin: float, sampler_inbatch: si.InBatchTripletsSampler
    ):
        """
        Args:
            margin: margin value
            sampler_inbatch: sampler for forming triplets inside the batch
        """
        super().__init__()
        self._sampler_inbatch = sampler_inbatch
        self._triplet_margin_loss = TripletMarginLoss(margin=margin)

    @staticmethod
    def _prepate_labels(labels: Union[Tensor, List[int]]) -> List[int]:
        """
        This function allows to work with 2 types of indexing:
        using a integer tensor and a list of indices.

        Args:
            labels: labels of batch samples

        Returns:
            labels of batch samples in the aligned format
        """
        if isinstance(labels, Tensor):
            labels = labels.squeeze()
            assert (labels.ndim == 1) and (
                labels.dtype in [short, tint, long]
            ), "Labels cannot be interpreted as indices."
            labels_list = labels.tolist()

        elif isinstance(labels, list):
            labels_list = labels.copy()

        else:
            raise TypeError(f"Unexpected type of labels: {type(labels)}).")

        return labels_list

    def forward(
        self, features: Tensor, labels: Union[Tensor, List[int]]
    ) -> Tensor:
        """
        Args:
            features: features with the shape of [batch_size, features_dim]
            labels: labels of samples having batch_size elements

        Returns: loss value

        """
        labels_list = self._prepate_labels(labels)

        features_a, features_p, features_n = self._sampler_inbatch.sample(
            features=features, labels=labels_list
        )
        loss = self._triplet_margin_loss(
            anchor=features_a, positive=features_p, negative=features_n
        )
        return loss


def run_ml_train() -> None:
    """
    Draft of Metric Learning training run_ml_train.
    By and large, to adapt this code for your task,
    all you have to do is prepare your dataset in
    the required format and select hyperparameters.
    """
    # data
    dataset = OmniglotML(
        root="/Users/alexeyshab/Downloads/",
        download=True,
        transform=t.ToTensor(),
    )
    sampler = BalanceBatchSampler(labels=dataset.get_labels(), p=32, k=4)
    train_loader = DataLoader(
        dataset=dataset, sampler=sampler, batch_size=sampler.batch_size
    )

    # model
    model = resnet18()
    model = adopt_resnet_for_1channel(model)
    optimizer = Adam(model.parameters(), lr=1e-3)

    # criterion
    # todo: normalisation
    # sampler_inbatch = si.AllTripletsSampler(max_output_triplets=512)
    sampler_inbatch = si.HardTripletsSampler(need_norm=False)
    criterion = TripletMarginLossWithSampling(
        margin=0.5, sampler_inbatch=sampler_inbatch
    )

    # train
    runner = SupervisedRunner()
    runner.train(
        model=model,
        criterion=criterion,
        optimizer=optimizer,
        loaders={"train": train_loader},
        num_epochs=5,
        verbose=True,
    )


run_ml_train()
