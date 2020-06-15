from collections import defaultdict
import os

import numpy as np

from catalyst.core import Callback, CallbackOrder, IRunner


# @TODO: refactor
class InferCallback(Callback):
    """@TODO: Docs. Contribution is welcome."""

    def __init__(self, out_dir=None, out_prefix=None):
        """
        Args:
            @TODO: Docs. Contribution is welcome
        """
        super().__init__(CallbackOrder.internal)
        self.out_dir = out_dir
        self.out_prefix = out_prefix
        self.predictions = defaultdict(lambda: [])
        self._keys_from_runner = ["out_dir", "out_prefix"]

    def on_stage_start(self, runner: IRunner):
        """Stage start hook.

        Args:
            runner (IRunner): current runner
        """
        for key in self._keys_from_runner:
            value = getattr(runner, key, None)
            if value is not None:
                setattr(self, key, value)
        # assert self.out_prefix is not None
        if self.out_dir is not None:
            self.out_prefix = f"{str(self.out_dir)}/{str(self.out_prefix)}"
        if self.out_prefix is not None:
            os.makedirs(os.path.dirname(self.out_prefix), exist_ok=True)

    def on_loader_start(self, runner: IRunner):
        """Loader start hook.

        Args:
            runner (IRunner): current runner
        """
        self.predictions = defaultdict(lambda: [])

    def on_batch_end(self, runner: IRunner):
        """Batch end hook.

        Args:
            runner (IRunner): current runner
        """
        dct = runner.output
        dct = {key: value.detach().cpu().numpy() for key, value in dct.items()}
        for key, value in dct.items():
            self.predictions[key].append(value)

    def on_loader_end(self, runner: IRunner):
        """Loader end hook.

        Args:
            runner (IRunner): current runner
        """
        self.predictions = {
            key: np.concatenate(value, axis=0)
            for key, value in self.predictions.items()
        }
        if self.out_prefix is not None:
            for key, value in self.predictions.items():
                suffix = ".".join([runner.loader_name, key])
                np.save(f"{self.out_prefix}/{suffix}.npy", value)


__all__ = ["InferCallback"]
