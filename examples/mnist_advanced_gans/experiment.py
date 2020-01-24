from collections import OrderedDict

from data_utils import SameClassBatchSampler

import torchvision
from torch.utils.data import DataLoader

from catalyst.dl import ConfigExperiment
from catalyst.utils import merge_dicts


class MNIST(torchvision.datasets.MNIST):

    def __init__(self, root, train=True, transform=None, target_transform=None,
                 download=False, image_key="image", target_key="target"):
        super().__init__(root, train, transform, target_transform, download)
        self.image_key = image_key
        self.target_key = target_key

    def __getitem__(self, index: int):
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

    def get_datasets(self, stage: str,
                     image_key: str = "image",
                     target_key: str = "target"):
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


# data loaders & transforms
class DAGANMnistGanExperiment(MnistGanExperiment):
    """
    Experiment with image conditioning
    (special batch sampler is used to not override the dataset)
    """
    def get_loaders(self, stage: str) -> "OrderedDict[str, DataLoader]":
        """

        :param stage:
        :return:
        """
        # WARN: 1)for simplicity of implementation `get_datasets`
        # will be called twice
        # WARN: 2)"loaders_params" are ignored in fact
        data_params = self.stages_config[stage]["data_params"]
        if "use_same_class_batch_sampler" in data_params:
            pass
        build_batch_sampler = data_params.get(
            "use_same_class_batch_sampler", False
        )
        if build_batch_sampler:
            batch_size = data_params.pop("batch_size", 1)
            drop_last = data_params.pop("drop_last", False)
            drop_odd_class_elements = data_params.pop(
                "drop_odd_class_elements", False
            )

            datasets = self.get_datasets(stage, **data_params)
            loaders_params_key = "loaders_params"
            data_params[loaders_params_key] = merge_dicts(
                {
                    dataset_name: {
                        "batch_sampler": SameClassBatchSampler(
                            dataset,
                            batch_size=batch_size,
                            drop_last_batch=drop_last,
                            drop_odd_class_elements=drop_odd_class_elements,
                            shuffle=dataset_name.startswith("train")
                        )
                    }
                    for dataset_name, dataset in datasets.items()
                }, data_params.get(loaders_params_key, {})
            )
        return super().get_loaders(stage)
