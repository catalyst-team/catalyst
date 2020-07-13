from typing import List
from abc import ABC, abstractmethod
from collections import OrderedDict
import os
from pathlib import Path
import shutil

import numpy as np
from PIL import Image

import torch
from torch import nn
from torch.nn.functional import relu
from torch.optim import Adam
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.datasets import Omniglot

from catalyst.contrib.data.ml import QueryGalleryDataset
from catalyst.contrib.data.transforms import ToTensor
from catalyst.contrib.datasets import MNIST
from catalyst.contrib.dl.callbacks import CMCScoreCallback
from catalyst.contrib.nn.criterion import TripletLoss
from catalyst.contrib.nn.modules.common import Normalize
from catalyst.dl.callbacks import ControlFlowCallback
from catalyst.dl.runner import SupervisedRunner


def create_splits_for_mnist(path: str, num_splits=1):
    path = Path(path)
    for i in range(num_splits):
        dataset = MNIST(
            os.getcwd(), train=True, download=True, transform=ToTensor()
        )
        split_dir = path / f"split_{i}"
        os.mkdir(split_dir)
        for cur_label in range(10):
            if cur_label == 0:
                query_path = split_dir / "query"
                gallery_path = split_dir / "gallery"
                os.mkdir(query_path)
                os.mkdir(gallery_path)
            query_obj_path = query_path / (str(cur_label))
            gallery_obj_path = gallery_path / (str(cur_label))
            os.mkdir(query_obj_path)
            os.mkdir(gallery_obj_path)
            idxs = torch.where(dataset.targets == cur_label)[0]
            query_idx = idxs[2 * i]
            gallery_idx = idxs[2 * i + 1]
            query_img = Image.fromarray(dataset.data[query_idx].numpy())
            gallery_img = Image.fromarray(dataset.data[gallery_idx].numpy())
            query_img.save(query_obj_path / "1.png", fromat="png")
            gallery_img.save(gallery_obj_path / "1.png", fromat="png")
        return path


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


class QueryGalleryFolderDataset(QueryGalleryDataset):
    def __init__(self, path: str, transform=None):
        path = Path(path)
        query_path = path / "query"
        gallery_path = path / "gallery"
        self.query_labels = []
        self.query_imgs = []
        for label in os.listdir(query_path):
            img_path = query_path / str(label)
            for img in os.listdir(img_path):
                pil_image = Image.open(img_path / img)
                self.query_imgs.append(pil_image)
                self.query_labels.append(int(label))
        self.gallery_labels = []
        self.gallery_imgs = []
        for label in os.listdir(gallery_path):
            img_path = query_path / str(label)
            for img in os.listdir(img_path):
                pil_image = Image.open(img_path / img)
                self.gallery_imgs.append(pil_image)
                self.gallery_labels.append(int(label))
        if transform is None:
            transform = transforms.ToTensor()
        self.transform = transform

    def __len__(self):
        return len(self.query_labels) + len(self.gallery_labels)

    @property
    def query_size(self):
        return len(self.query_labels)

    @property
    def gallery_size(self):
        return len(self.gallery_labels)

    def __getitem__(self, index):
        if index >= self.query_size:
            index -= self.query_size
            return {
                "features": self.transform(self.gallery_imgs[index]),
                "targets": self.gallery_labels[index],
                "is_query": False,
            }
        return {
            "features": self.transform(self.query_imgs[index]),
            "targets": 0,  # not a typo, for test needs
            "is_query": True,
        }


def test_pipeline():
    """Test if simple pipeline works"""
    create_splits_for_mnist("tmp")
    dataset = MnistML(
        root="tmp",
        train=False,
        download=True,
        transform=transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
        ),
    )
    val_dataset = MnistML(
        root="tmp",
        train=False,
        download=True,
        transform=transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
        ),
    )

    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=32)
    qg_dataset = QueryGalleryFolderDataset("split_0")
    qg_dataloader = DataLoader(qg_dataset, batch_size=32)
    loaders = {
        "train": dataloader,
        "valid": val_dataloader,
        "query/gallery": qg_dataloader,
    }

    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc1 = nn.Linear(768, 2)
            self.norm = Normalize()

        def forward(self, x):
            x = torch.flatten(x, 1)
            x = relu(self.fc1(x))
            return x

    model = Net()
    optimizer = Adam(model.parameters())
    criterion = TripletLoss()

    runner = SupervisedRunner()
    runner.train(
        model=model,
        criterion=criterion,
        optimizer=optimizer,
        loaders=OrderedDict(loaders),
        callbacks=[
            ControlFlowCallback(
                base_callback=CMCScoreCallback(topk_args=[1, 5],),
                loaders=["query/gallery"],
            )
        ],
        num_epochs=3,
        verbose=True,
    )
    shutil.rmtree("tmp")
    assert np.isclose(runner.loader_metrics["cmc_1"], 0.1)
    assert np.isclose(runner.loader_metrics["cmc_5"], 0.1)
