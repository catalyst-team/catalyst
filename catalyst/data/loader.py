from typing import Union

from torch.utils.data import DataLoader


class BatchLimitLoaderWrapper:
    """
    Loader wrapper. Limits number of batches used per each iteration.

    For example, if you have some loader and want to use only first 5 bathes:

    .. code-block:: python

        import torch
        from torch.utils.data import DataLoader, TensorDataset
        from catalyst.data.loader import BatchLimitLoaderWrapper

        num_samples, num_features = int(1e4), int(1e1)
        X, y = torch.rand(num_samples, num_features), torch.rand(num_samples)
        dataset = TensorDataset(X, y)
        loader = DataLoader(dataset, batch_size=32, num_workers=1)
        loader = BatchLimitLoaderWrapper(loader, num_batches=5)

    or if you would like to use only some portion of Dataloader
    (we use 30% in the example below):

    .. code-block:: python

        import torch
        from torch.utils.data import DataLoader, TensorDataset
        from catalyst.data.loader import BatchLimitLoaderWrapper

        num_samples, num_features = int(1e4), int(1e1)
        X, y = torch.rand(num_samples, num_features), torch.rand(num_samples)
        dataset = TensorDataset(X, y)
        loader = DataLoader(dataset, batch_size=32, num_workers=1)
        loader = BatchLimitLoaderWrapper(loader, num_batches=0.3)

    .. note::
        Generally speaking, this wrapper could be used with any iterator-like
        object. No ``DataLoader``-specific code used.
    """

    def __init__(self, loader: DataLoader, num_batches: Union[int, float]):
        """
        Loader wrapper. Limits number of batches used per each iteration.

        Args:
            loader (DataLoader): torch dataloader.
            num_batches (Union[int, float]): number of batches to use (int),
                or portion of iterator (float, should be in [0;1] range)
        """
        assert isinstance(num_batches, (int, float)), (
            "Expected ``num_batches`` type is int/float"
            f"but got {type(num_batches)}"
        )
        if isinstance(num_batches, float):
            assert 0.0 <= num_batches <= 1, (
                "Expected ``num_batches`` to be in range [0; 1]"
                f"but got {num_batches}"
            )
            num_batches = int(len(loader) * num_batches)

        self.origin = loader
        self.iterator = iter(self.origin)
        self.iteration_index = 0
        self.num_batches = num_batches

    def __getattr__(self, key):
        """
        Gets attribute by ``key``.
        Firstly, looks at the ``origin`` for the appropriate ``key``.
        If none founds - looks at the wrappers attributes.
        If could not found anything - raises ``NotImplementedError``.

        Args:
            key: attribute key

        Returns:
            attribute value

        Raises:
            NotImplementedError: if could not find attribute in ``origin``
                or ``wrapper``
        """
        value = getattr(self.origin, key, None)
        if value is not None:
            return value
        value = getattr(self, key, None)
        if value is not None:
            return value
        raise NotImplementedError()

    def __iter__(self):
        """Iterator.

        Returns:
            iterator object
        """
        self.iteration_index = 0
        self.iterator = iter(self.origin)
        return self

    def __next__(self):
        """Next batch.

        Returns:
            next batch
        """
        if self.iteration_index >= len(self.origin):
            raise StopIteration()
        self.iteration_index += 1
        if self.iteration_index % self.num_batches == 0:
            self.iterator = iter(self.origin)
        batch = next(self.iterator)
        return batch

    def __len__(self) -> int:
        """Returns length of the wrapper loader.

        Returns:
            int: length of the wrapper loader
        """
        return len(self.origin)


__all__ = ["BatchLimitLoaderWrapper"]
