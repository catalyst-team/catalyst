from collections import OrderedDict

from data_utils import SameClassBatchSampler

import torchvision
from torchvision import transforms
from torch.utils.data import DataLoader

from catalyst.dl import ConfigExperiment
from catalyst.utils import merge_dicts


# data loaders & transforms
class MnistGanExperiment(ConfigExperiment):
    """
    Simple MNIST experiment
    """
    @staticmethod
    def get_transforms(stage: str = None, mode: str = None):
        """

        :param stage:
        :param mode:
        :return:
        """
        return transforms.Compose(
            [transforms.ToTensor(),
             transforms.Normalize((0.5, ), (0.5, ))]
        )

    def get_datasets(self, stage: str, **kwargs):
        """

        :param stage:
        :param kwargs:
        :return:
        """
        datasets = OrderedDict()

        trainset = torchvision.datasets.MNIST(
            root="./data",
            train=True,
            download=True,
            transform=MnistGanExperiment.get_transforms(
                stage=stage, mode="train"
            )
        )
        testset = torchvision.datasets.MNIST(
            root="./data",
            train=False,
            download=True,
            transform=MnistGanExperiment.get_transforms(
                stage=stage, mode="valid"
            )
        )

        datasets["train"] = trainset
        datasets["valid"] = testset

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
