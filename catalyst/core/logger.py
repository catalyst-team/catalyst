from typing import Dict

import numpy as np


# @TODO: could we just pass runner?
class ILogger:
    """An abstraction that syncs experiment run with monitoring tools."""

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
        pass

    def log_image(
        self,
        image: np.ndarray,
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
        pass

    def log_hparams(
        self,
        hparams: Dict,
        scope: str = None,
        # experiment info
        experiment_key: str = None,
    ) -> None:
        pass

    def flush_log(self) -> None:
        pass

    def close_log(self) -> None:
        pass


__all__ = ["ILogger"]
