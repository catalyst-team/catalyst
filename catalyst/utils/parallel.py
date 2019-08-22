# description     :Multiprocessing and tqdm wrapper for easy paralelizing
# author          :Vsevolod Poletaev
# author_email    :poletaev.va@gmail.com
# date            :20190822
# version         :19.08.7
# ==============================================================================

from typing import List, Union, TypeVar
from multiprocessing.pool import Pool

from tqdm import tqdm

T = TypeVar("T")


class DumbPool:
    def imap_unordered(self, func, args):
        return map(func, args)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        return self


def parallel_imap(
    func,
    args,
    pool: Union[Pool, DumbPool],
) -> List[T]:
    result = list(pool.imap_unordered(func, args))
    return result


def tqdm_parallel_imap(
    func,
    args,
    pool: Union[Pool, DumbPool],
    total: int = None,
    pbar=tqdm,
) -> List[T]:
    if total is None and hasattr(args, "__len__"):
        total = len(args)

    if pbar is None:
        result = parallel_imap(func, args, pool)
    else:
        result = list(pbar(pool.imap_unordered(func, args), total=total))

    return result


def get_pool(workers: int) -> Union[Pool, DumbPool]:
    pool = Pool(workers) if workers > 0 and workers is not None else DumbPool()
    return pool
