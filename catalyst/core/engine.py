# taken from https://github.com/Scitator/animus/blob/main/animus/torch/accelerate.py
from typing import Callable, Dict

from accelerate import Accelerator


class IEngine(Accelerator):
    @staticmethod
    def spawn(fn: Callable):
        fn()

    @staticmethod
    def setup(local_rank: int, world_size: int):
        pass

    @staticmethod
    def cleanup():
        pass

    def mean_reduce_ddp_metrics(self, metrics: Dict):
        pass


__all__ = ["IEngine"]
