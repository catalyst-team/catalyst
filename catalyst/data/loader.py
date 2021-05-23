from typing import Any, Callable, Iterable, Union
from itertools import tee
import queue
import sys
import threading

import numpy as np
import torch
from torch.utils.data import DataLoader


class ILoaderWrapper:
    """Loader wrapper interface.

    Args:
        loader: torch dataloader.
    """

    def __init__(self, loader: DataLoader):
        """Init"""
        self.origin = loader

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

    def __len__(self) -> int:
        """Returns length of the wrapper loader.

        Returns:
            int: length of the wrapper loader
        """
        return len(self.origin)


class BatchLimitLoaderWrapper(ILoaderWrapper):
    """Loader wrapper. Limits number of batches used per each iteration.

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
        """Loader wrapper. Limits number of batches used per each iteration.

        Args:
            loader: torch dataloader.
            num_batches (Union[int, float]): number of batches to use (int),
                or portion of iterator (float, should be in [0;1] range)
        """
        super().__init__(loader)
        assert isinstance(num_batches, (int, float)), (
            "Expected ``num_batches`` type is int/float" f"but got {type(num_batches)}"
        )
        if isinstance(num_batches, float):
            assert 0.0 <= num_batches <= 1, (
                "Expected ``num_batches`` to be in range [0; 1]" f"but got {num_batches}"
            )
            num_batches = int(len(loader) * num_batches)

        self._iterator = iter(self.origin)
        self.iteration_index = 0
        self.num_batches = num_batches

    def __iter__(self):
        """Iterator.

        Returns:
            iterator object
        """
        self.iteration_index = 0
        self._iterator, self.iterator = tee(self._iterator)
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
            self._iterator, self.iterator = tee(self._iterator)
        batch = next(self.iterator)
        return batch


def _any2cuda_non_blocking(value: Any):
    # based on catalyst.utils.torch.any2device
    # but with cuda non_blocking trick
    if isinstance(value, dict):
        return {k: _any2cuda_non_blocking(v) for k, v in value.items()}
    elif isinstance(value, (tuple, list)):
        return [_any2cuda_non_blocking(v) for v in value]
    elif torch.is_tensor(value):
        return value.cuda(non_blocking=True)
    elif isinstance(value, (np.ndarray, np.void)) and value.dtype.fields is not None:
        return {k: _any2cuda_non_blocking(value[k]) for k in value.dtype.fields.keys()}
    elif isinstance(value, np.ndarray):
        return torch.tensor(value).cuda(non_blocking=True)


def _map_loop(
    func: Callable,
    iterable: Iterable,
    result_queue: queue.Queue,
    error_queue: queue.Queue,
    done_event: threading.Event,
):
    try:
        for x in iterable:
            result = func(x)
            result_queue.put(result)
    except BaseException:  # noqa: WPS424
        error_queue.put(sys.exc_info())
    finally:
        done_event.set()


def _prefetch_map(
    func: Callable, iterable: Iterable, num_prefetches: int = 1, timeout: int = 2,
) -> Iterable:
    result_queue = queue.Queue(num_prefetches)
    error_queue = queue.Queue(1)
    done_event = threading.Event()
    map_thread = threading.Thread(
        target=_map_loop, args=(func, iterable, result_queue, error_queue, done_event),
    )
    map_thread.daemon = True
    map_thread.start()
    while not (done_event.is_set() and result_queue.empty()):
        try:
            result = result_queue.get(timeout=timeout)
        except queue.Empty:
            continue
        yield result
    if error_queue.full():
        raise error_queue.get()[1]


def _prefetch_loader(loader: DataLoader, num_prefetches: int) -> Iterable:
    if torch.cuda.is_available():
        return _prefetch_map(_any2cuda_non_blocking, loader, num_prefetches=num_prefetches)
    else:
        return iter(loader)


class BatchPrefetchLoaderWrapper(ILoaderWrapper):
    """Loader wrapper. Prefetches specified number of batches on the GPU.

    Base usage:

    .. code-block:: python

        import torch
        from torch.utils.data import DataLoader, TensorDataset
        from catalyst.data import BatchPrefetchLoaderWrapper

        num_samples, num_features = int(1e4), int(1e1)
        X, y = torch.rand(num_samples, num_features), torch.rand(num_samples)
        dataset = TensorDataset(X, y)
        loader = DataLoader(dataset, batch_size=32, num_workers=1)
        loader = BatchPrefetchLoaderWrapper(loader)

    Minimal working example:

    .. code-block:: python

        import os
        import torch
        from torch.nn import functional as F
        from torch.utils.data import DataLoader
        from catalyst import dl, metrics
        from catalyst.data.cv import ToTensor
        from catalyst.contrib.datasets import MNIST
        from catalyst.data import BatchPrefetchLoaderWrapper

        class CustomRunner(dl.Runner):

            def handle_batch(self, batch):
                # model train/valid step
                x, y = batch
                y_hat = self.model(x.view(x.size(0), -1))

                loss = F.cross_entropy(y_hat, y)
                accuracy01 = metrics.accuracy(y_hat, y, topk=(1, ))
                self.batch_metrics.update(
                    {"loss": loss, "accuracy01": accuracy01}
                )

                if self.is_train_loader:
                    loss.backward()
                    self.optimizer.step()
                    self.optimizer.zero_grad()

        model = torch.nn.Linear(28 * 28, 10)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.02)

        batch_size=32
        loaders = {
            "train": DataLoader(
                MNIST(
                    os.getcwd(),
                    train=True,
                    download=True,
                    transform=ToTensor()
                ),
                batch_size=batch_size),
            "valid": DataLoader(
                MNIST(
                    os.getcwd(),
                    train=False,
                    download=True,
                    transform=ToTensor()
                ),
                batch_size=batch_size),
        }
        loaders = {
            k: BatchPrefetchLoaderWrapper(v) for k, v in loaders.items()
        }

        runner = CustomRunner()
        # model training
        runner.train(
            model=model,
            optimizer=optimizer,
            loaders=loaders,
            logdir="./logs",
            num_epochs=5,
            verbose=True,
            load_best_on_end=True,
        )

    """

    def __init__(self, loader: DataLoader, num_prefetches: int = None):
        """Loader wrapper. Prefetches specified number of batches on the GPU.

        Args:
            loader: torch dataloader.
            num_prefetches: number of batches to prefetch on the GPU.
        """
        super().__init__(loader)
        self.num_prefetches = num_prefetches or 1

    def __iter__(self):
        """Iterator.

        Returns:
            iterator object
        """
        return _prefetch_loader(self.origin, self.num_prefetches)


__all__ = ["ILoaderWrapper", "BatchLimitLoaderWrapper", "BatchPrefetchLoaderWrapper"]
