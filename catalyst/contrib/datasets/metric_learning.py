from abc import ABC, abstractmethod
from typing import List

from torch.utils.data import Dataset
from catalyst.contrib.datasets import MNIST
from torchvision.datasets import Omniglot


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
        raise NotImplementedError


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


class MnistML(MetricLearningTrainDataset, MNIST):
    """
    Simple wrapper for MNIST dataset
    """

    def get_labels(self) -> List[int]:
        """
        Returns:
            labels of digits
        """
        return self.targets.tolist()
