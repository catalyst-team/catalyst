from torch.utils.data import BatchSampler, DataLoader, IterableDataset

# kwargs of the DataLoader in min version 1.3.0.
_PYTORCH_DATALOADER_KWARGS = {
    "batch_size": 1,
    "shuffle": False,
    "sampler": None,
    "batch_sampler": None,
    "num_workers": 0,
    "collate_fn": None,
    "pin_memory": False,
    "drop_last": False,
    "timeout": 0,
    "worker_init_fn": None,
    "multiprocessing_context": None,
}


# Heavily based on HuggingFace accelerate project internals.
# https://github.com/huggingface/accelerate/blob/main/src/accelerate/data_loader.py
class BatchSamplerShard(BatchSampler):
    """
    Wraps a PyTorch :obj:`BatchSampler` to generate batches for one of the processes only.
    Instances of this class will always yield a number of batches that is a round multiple
    of :obj:`num_processes` and that all have the same size.
    Depending on the value of the :obj:`drop_last` attribute of the batch sampler passed,
    it will either stop the iteration at the first batch that would be too small /
    not present on all processes or loop with indices from the beginning.

    Args:
        batch_sampler (:obj:`torch.utils.data.sampler.BatchSampler`):
            The batch sampler to split in several shards.
        num_processes (:obj:`int`, `optional`, defaults to 1):
            The number of processes running concurrently.
        process_index (:obj:`int`, `optional`, defaults to 0):
            The index of the current process.

    .. warning::

        This does not support :obj:`BatchSampler` with varying batch size yet.
    """

    def __init__(
        self,
        batch_sampler: BatchSampler,
        num_processes: int = 1,
        process_index: int = 0,
    ):
        """Init."""
        self.batch_sampler = batch_sampler
        self.num_processes = num_processes
        self.process_index = process_index
        self.batch_size = batch_sampler.batch_size
        self.drop_last = batch_sampler.drop_last

    def __len__(self):
        """Docs."""
        if len(self.batch_sampler) % self.num_processes == 0:
            return len(self.batch_sampler) // self.num_processes
        length = len(self.batch_sampler) // self.num_processes
        return length if self.drop_last else length + 1

    def __iter__(self):
        """Docs."""
        initial_data = []
        batch_to_yield = []
        for idx, batch in enumerate(self.batch_sampler):
            # We gather the initial indices in case we need to circle back at the end.
            if not self.drop_last and idx < self.num_processes:
                initial_data += batch
            # We identify the batch to yield
            # but wait until we are sure every process gets a full batch
            # before actually yielding it.
            if idx % self.num_processes == self.process_index:
                batch_to_yield = batch
            if (
                idx % self.num_processes == self.num_processes - 1
                and len(batch) == self.batch_size
            ):
                yield batch_to_yield
                batch_to_yield = []

        # If drop_last is True, iteration is over, otherwise...
        if not self.drop_last and len(initial_data) > 0:
            # ... we yield the complete batch we had saved before if it has the proper length
            if len(batch_to_yield) == self.batch_size:
                yield batch_to_yield

            # For degenerate cases where the dataset has less than num_process * batch_size samples
            while len(initial_data) < self.num_processes * self.batch_size:
                initial_data += initial_data

            # If the last batch seen was of the proper size,
            # it has been yielded by its process so we move to the next
            if len(batch) == self.batch_size:
                batch = []
                idx += 1

            # Make sure we yield a multiple of self.num_processes batches
            cycle_index = 0
            while idx % self.num_processes != 0 or len(batch) > 0:
                end_index = cycle_index + self.batch_size - len(batch)
                batch += initial_data[cycle_index:end_index]
                if idx % self.num_processes == self.process_index:
                    yield batch
                cycle_index = end_index
                batch = []
                idx += 1


def prepare_ddp_loader(loader: DataLoader, num_processes: int, process_index: int) -> DataLoader:
    """
    Transfers loader to distributed mode. Experimental feature.

    Args:
        loader: pytorch dataloder
        num_processes (:obj:`int`, `optional`, defaults to 1):
            The number of processes running concurrently.
        process_index (:obj:`int`, `optional`, defaults to 0):
            The index of the current process.

    Returns:
        DataLoader: pytorch dataloder with distributed batch sampler.
    """
    ddp_dataset = loader.dataset
    # Iterable dataset doesn't like batch_sampler, but DataLoader creates a default one for it
    if isinstance(ddp_dataset, IterableDataset):
        ddp_batch_sampler = None
    else:
        ddp_batch_sampler = BatchSamplerShard(
            loader.batch_sampler,
            num_processes=num_processes,
            process_index=process_index,
        )
    # We ignore all of those since they are all dealt with by our new_batch_sampler
    ignore_kwargs = [
        "batch_size",
        "shuffle",
        "sampler",
        "batch_sampler",
        "drop_last",
        "generator",
    ]
    kwargs = {
        k: getattr(loader, k, _PYTORCH_DATALOADER_KWARGS[k])
        for k in _PYTORCH_DATALOADER_KWARGS
        if k not in ignore_kwargs
    }
    # Need to provide batch_size as batch_sampler is None for Iterable dataset
    if ddp_batch_sampler is None:
        kwargs["drop_last"] = loader.drop_last
        kwargs["batch_size"] = loader.batch_size

    loader = DataLoader(dataset=ddp_dataset, batch_sampler=ddp_batch_sampler, **kwargs)
    return loader


__all__ = [BatchSamplerShard, prepare_ddp_loader]
