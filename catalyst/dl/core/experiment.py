from typing import Union  # isort:skip

from catalyst.core import _Experiment


class Experiment(_Experiment):
    def get_native_batch(
        self, stage: str, loader: Union[str, int] = 0, data_index: int = 0
    ):
        """Returns a batch from experiment loader

        Args:
            stage (str): stage name
            loader (Union[str, int]): loader name or its index,
                default is the first loader
            data_index (int): index in dataset from the loader
        """
        loaders = self.get_loaders(stage)
        if isinstance(loader, str):
            _loader = loaders[loader]
        elif isinstance(loader, int):
            _loader = list(loaders.values())[loader]
        else:
            raise TypeError("Loader parameter must be a string or an integer")

        dataset = _loader.dataset
        collate_fn = _loader.collate_fn

        sample = collate_fn([dataset[data_index]])

        return sample


__all__ = ["Experiment"]
