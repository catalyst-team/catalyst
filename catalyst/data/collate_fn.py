import collections
from torch.utils.data.dataloader import default_collate


class FilteringCollateFn:
    """
    Callable object doing job of ``collate_fn`` like ``default_collate``,
    but does not cast batch items with specified key to :class:`torch.Tensor`.

    Only adds them to list.
    Supports only key-value format batches
    """
    def __init__(self, *keys):
        """
        Args:
            keys: Keys for values that will not be
                converted to tensor and stacked
        """

        self.keys = keys

    def __call__(self, batch):
        """
        Args:
            batch: current batch
        Returns:
            batch values filtered by `keys`
        """
        if isinstance(batch[0], collections.Mapping):
            result = {}
            for key in batch[0]:
                items = [d[key] for d in batch]
                if key not in self.keys:
                    items = default_collate(items)
                result[key] = items
            return result
        else:
            return default_collate(batch)


__all__ = ["FilteringCollateFn"]
