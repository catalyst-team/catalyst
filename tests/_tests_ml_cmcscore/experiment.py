# flake8: noqa
from collections import OrderedDict
import os
from pathlib import Path
import shutil

from PIL import Image

import torch

from catalyst.contrib.data.ml import QueryGalleryFolderDataset
from catalyst.contrib.data.transforms import Compose, Normalize, ToTensor
from catalyst.contrib.datasets import MNIST
from catalyst.dl import ConfigExperiment


class QueryGalleryFolderDatasetCorrupted(QueryGalleryFolderDataset):
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


def create_splits_for_mnist(path: str, num_splits=1):
    path = Path(path)
    for i in range(num_splits):
        dataset = MNIST(
            "./data", train=True, download=True, transform=ToTensor()
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


class Experiment(ConfigExperiment):
    """
    @TODO: Docs. Contribution is welcome
    """

    @staticmethod
    def get_transforms(stage: str = None, mode: str = None):
        """
        @TODO: Docs. Contribution is welcome
        """
        return Compose([ToTensor(), Normalize((0.1307,), (0.3081,))])

    def get_datasets(self, stage: str, **kwargs):
        """
        @TODO: Docs. Contribution is welcome
        """
        datasets = OrderedDict()
        shutil.rmtree("./data/split_0", ignore_errors=True)
        create_splits_for_mnist("./data")

        trainset = MNIST(
            "./data",
            train=False,
            download=True,
            transform=Experiment.get_transforms(stage=stage, mode="train"),
        )
        testset = MNIST(
            "./data",
            train=False,
            download=True,
            transform=Experiment.get_transforms(stage=stage, mode="valid"),
        )

        datasets["train"] = trainset
        datasets["valid"] = testset
        if stage == "corrupted":
            datasets["query_gallery"] = QueryGalleryFolderDatasetCorrupted(
                "./data/split_0"
            )
        else:
            datasets["query_gallery"] = QueryGalleryFolderDataset(
                "./data/split_0"
            )

        return datasets
