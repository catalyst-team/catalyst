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


class CustomSupervisedConfigRunner(IRunnerMixin, SupervisedConfigRunner):
    def _get_dataset_from_params(self, num_samples_per_class=320, **kwargs):
        dataset = super()._get_dataset_from_params(**kwargs)
        if kwargs.get("train", True):
            dataset = {
                "dataset": dataset,
                "sampler": BalanceClassSampler(labels=dataset.targets, mode=num_samples_per_class),
            }

        return dataset


if SETTINGS.hydra_required:
    import hydra

    from catalyst.dl import SupervisedHydraRunner

    class CustomSupervisedHydraRunner(IRunnerMixin, SupervisedHydraRunner):
        def _get_dataset_from_params(self, params):
            num_samples_per_class = params.pop("num_samples_per_class", 320)

            dataset = super()._get_dataset_from_params(params)
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
