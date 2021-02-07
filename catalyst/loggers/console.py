from typing import Any, Dict
import os

import numpy as np

from catalyst.contrib.tools.tensorboard import SummaryWriter
from catalyst.core.logger import ILogger
from catalyst.utils.config import save_config


def _format_metrics(dct: Dict):
    return " | ".join([f"{k}: {dct[k]}" for k in sorted(dct.keys())])


class ConsoleLogger(ILogger):
    def log_metrics(
        self,
        metrics: Dict[str, float],
        scope: str = None,
        # experiment info
        experiment_key: str = None,
        global_epoch_step: int = 0,
        global_batch_step: int = 0,
        global_sample_step: int = 0,
        # stage info
        stage_key: str = None,
        stage_epoch_len: int = 0,
        stage_epoch_step: int = 0,
        stage_batch_step: int = 0,
        stage_sample_step: int = 0,
        # loader info
        loader_key: str = None,
        loader_batch_len: int = 0,
        loader_sample_len: int = 0,
        loader_batch_step: int = 0,
        loader_sample_step: int = 0,
    ) -> None:
        # if self.exclude is not None and scope in self.exclude:
        #     return
        # elif (
        #     self.include is not None and scope in self.include
        # ) or self.include is None:
        if scope == "loader":
            prefix = f"{loader_key} ({stage_epoch_step}/{stage_epoch_len}) "
            msg = prefix + _format_metrics(metrics)
            print(msg)
        elif scope == "epoch":
            # @TODO: trick to save pure epoch-based metrics, like lr/momentum
            prefix = f"* Epoch ({stage_epoch_step}/{stage_epoch_len}) "
            msg = prefix + _format_metrics(metrics["_epoch_"])
            print(msg)

    def log_hparams(
        self,
        hparams: Dict,
        scope: str = None,
        # experiment info
        experiment_key: str = None,
    ) -> None:
        print(f"Hparams ({experiment_key}): {hparams}")


__all__ = ["ConsoleLogger"]
