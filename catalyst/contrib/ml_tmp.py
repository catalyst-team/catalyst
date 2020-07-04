from abc import ABC, abstractmethod
from typing import List

import torchvision.transforms as t
from torch import nn, Tensor
from torch.nn import TripletMarginLoss
from torch.optim import Adam
from torch.utils.data import Dataset, DataLoader
from torchvision.datasets import Omniglot
from torchvision.models import resnet18

from catalyst.data.sampler import BalanceBatchSampler
from catalyst.data.sampler_inbatch import InBatchTripletsSampler, AllTripletsSampler
from catalyst.dl.runner import SupervisedRunner


class MetricLearningTrainDataset(Dataset, ABC):

    @abstractmethod
    def get_labels(self) -> List[int]:
        raise NotImplemented


class OmniglotML(MetricLearningTrainDataset, Omniglot):

    def get_labels(self) -> List[int]:
        return [label for _, label in self._flat_character_images]


class TripletMarginLossWithSampling(nn.Module):

    def __init__(self, sampler_inbatch: InBatchTripletsSampler):
        super().__init__()
        self._sampler_inbatch = sampler_inbatch
        self._triplet_margin_loss = TripletMarginLoss()

    def forward(self, features: Tensor, labels: List[int]) -> Tensor:
        print(features.shape, labels.shape)
        if isinstance(labels, Tensor):
            labels = labels.squeeze().tolist()

        features_a, features_p, features_n = self._sampler_inbatch.sample(labels=labels)
        loss = self._triplet_margin_loss(anchor=features_a, positive=features_p, negative=features_n)
        return loss


def pipeline():
    dataset = OmniglotML(root='/Users/alexeyshab/Downloads/', download=True,
                         transform=t.ToTensor())
    sampler = BalanceBatchSampler(labels=dataset.get_labels(), p=32, k=4)
    train_loader = DataLoader(dataset=dataset, sampler=sampler,
                              batch_size=sampler.batch_size)

    model = resnet18()
    model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2,
                            padding=3, bias=False)

    optimizer = Adam(model.parameters(), lr=1e-3)

    inbatch_sampler = AllTripletsSampler(max_output_triplets=512)
    criterion = TripletMarginLossWithSampling(inbatch_sampler)

    runner = SupervisedRunner()
    runner.train(model=model, criterion=criterion, optimizer=optimizer,
                 loaders={'train': train_loader}, num_epochs=1)


pipeline()
