from typing import Union

from torch.utils.data import DataLoader


class BatchLimitLoaderWrapper:
    def __init__(self, loader: DataLoader, num_batches: Union[int, float]):
        assert isinstance(num_batches, (int, float)), (
            "Expected loader num_batches type is int/float"
            f"but got {type(num_batches)}"
        )
        if isinstance(num_batches, float):
            num_batches = int(len(loader) * num_batches)

        self.loader = loader
        self.loader_iter = iter(self.loader)
        self.num_batches = num_batches
        self.iteration_index = 0

    def __getattr__(self, key):
        value = getattr(self.loader, key, None)
        if value is not None:
            return value
        value = getattr(self, key, None)
        if value is not None:
            return value
        raise NotImplementedError()

    def __iter__(self):
        self.iteration_index = 0
        return self

    def __next__(self):
        if self.iteration_index >= len(self.loader):
            raise StopIteration()
        self.iteration_index += 1
        if self.iteration_index % self.num_batches == 0:
            self.loader_iter = iter(self.loader)
        batch = next(self.loader_iter)
        return batch

    def __len__(self):
        return len(self.loader)
