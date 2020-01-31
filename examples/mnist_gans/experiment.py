from collections import OrderedDict

import torchvision

from catalyst.dl import ConfigExperiment


class MNIST(torchvision.datasets.MNIST):
    """
    MNIST Dataset with key_value __get_item__ output
    """
    def __init__(
        self,
        root,
        train=True,
        transform=None,
        target_transform=None,
        download=False,
        image_key="image",
        target_key="target"
    ):
        """

        :param root:
        :param train:
        :param transform:
        :param target_transform:
        :param download:
        :param image_key: key to place an image
        :param target_key: key to place target
        """
        super().__init__(root, train, transform, target_transform, download)
        self.image_key = image_key
        self.target_key = target_key

    def __getitem__(self, index: int):
        """Get dataset element"""
        image, target = self.data[index], self.targets[index]

        dict_ = {
            self.image_key: image,
            self.target_key: target,
        }

        if self.transform is not None:
            dict_ = self.transform(dict_)
        return dict_


# data loaders & transforms
class MnistGanExperiment(ConfigExperiment):
    """
    Simple MNIST experiment
    """
    def get_datasets(
        self, stage: str, image_key: str = "image", target_key: str = "target"
    ):
        """

        :param stage:
        :param image_key:
        :param target_key:
        :return:
        """
        datasets = OrderedDict()

        for dataset_name in ("train", "valid"):
            datasets[dataset_name] = MNIST(
                root="./data",
                train=(dataset_name == "train"),
                download=True,
                image_key=image_key,
                target_key=target_key,
                transform=self.get_transforms(
                    stage=stage, dataset=dataset_name
                )
            )

        return datasets
