# flake8: noqa
from collections import OrderedDict

from catalyst import utils
from catalyst.contrib.datasets import MNIST
from catalyst.data import ToTensor
from catalyst.data.sampler import BalanceClassSampler
from catalyst.dl import IRunner, SupervisedConfigRunner
from catalyst.settings import SETTINGS


class IRunnerMixin(IRunner):
    def get_model(self, stage: str):
        if self.model is None:
            # first stage
            model = super().get_model(stage=stage)
        else:
            model = self.model
        conv_layres = ["conv_net"]
        if stage == "tune":
            # second stage logic
            model = self.model
            for key in conv_layres:
                layer = getattr(model, key)
                utils.set_requires_grad(layer, requires_grad=False)
        return model

    def get_transform(self, stage: str = None, mode: str = None):
        return ToTensor()


class CustomSupervisedConfigRunner(IRunnerMixin, SupervisedConfigRunner):
    def get_dataset_from_params(
        self,
        root: str = "./data",
        train: bool = True,
        download: bool = False,
        num_samples_per_class=320,
    ):
        dataset = MNIST(root, train=train, download=download, transform=self.get_transform(),)
        if train:
            dataset = {
                "dataset": dataset,
                "sampler": BalanceClassSampler(labels=dataset.targets, mode=num_samples_per_class),
            }

        return dataset


if SETTINGS.hydra_required:
    import hydra

    from catalyst.dl import SupervisedHydraRunner

    class CustomSupervisedHydraRunner(IRunnerMixin, SupervisedHydraRunner):
        def get_dataset_from_params(self, params):
            num_samples_per_class = 320

            dataset = hydra.utils.instantiate(params, transform=self.get_transform())
            if params["train"]:
                dataset = {
                    "dataset": dataset,
                    "sampler": BalanceClassSampler(
                        labels=dataset.targets, mode=num_samples_per_class
                    ),
                }

            return dataset

    __all__ = ["CustomSupervisedConfigRunner", "CustomSupervisedHydraRunner"]
else:
    __all__ = ["CustomSupervisedConfigRunner"]
