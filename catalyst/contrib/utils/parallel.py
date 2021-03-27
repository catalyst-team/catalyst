# description     :Multiprocessing and tqdm wrapper for easy parallelizing
# author          :Vsevolod Poletaev
# author_email    :poletaev.va@gmail.com
# date            :20190822
# version         :19.08.7
# ==============================================================================
# flake8: noqa
# @TODO: code formatting issue for 20.07 release
from typing import List, TypeVar, Union
from multiprocessing.pool import Pool

# from tqdm import tqdm
from torch.utils.model_zoo import tqdm

T = TypeVar("T")


class DumbPool:
    """@TODO: Docs. Contribution is welcome."""

    def imap_unordered(self, func, args):
        """@TODO: Docs. Contribution is welcome."""
        return map(func, args)

    def __enter__(self):
        """Enter the runtime context related to ``DumbPool`` object."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit the runtime context related to ``DumbPool`` object."""
        return self


def parallel_imap(func, args, pool: Union[Pool, DumbPool]) -> List[T]:
    """@TODO: Docs. Contribution is welcome."""
    result = list(pool.imap_unordered(func, args))
    return result


def tqdm_parallel_imap(
    func, args, pool: Union[Pool, DumbPool], total: int = None, pbar=tqdm,
) -> List[T]:
    """@TODO: Docs. Contribution is welcome."""
    if total is None and hasattr(args, "__len__"):
        total = len(args)

    if pbar is None:
        result = parallel_imap(func, args, pool)
    else:
        result = list(pbar(pool.imap_unordered(func, args), total=total))

    return result


def get_pool(workers: int) -> Union[Pool, DumbPool]:
    """@TODO: Docs. Contribution is welcome."""
    pool = Pool(workers) if workers > 0 and workers is not None else DumbPool()
    return pool


__all__ = ["parallel_imap", "tqdm_parallel_imap", "get_pool"]
