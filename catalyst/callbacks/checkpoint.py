from typing import Any, Dict, List, Union
from collections import namedtuple
import json
import os
import shutil

import torch

from catalyst.core.callback import Callback, CallbackOrder
from catalyst.core.runner import IRunner

Checkpoint = namedtuple("Checkpoint", field_names=["obj", "logpath", "metric"])


class ICheckpointCallback(Callback):
    """Criterion callback interface, abstraction over checkpoint step."""

    pass


class CheckpointCallback(ICheckpointCallback):
    """Checkpoint callback to save/restore your model/runner."""

    def __init__(
        self,
        logdir: str,
        loader_key: str = None,
        metric_key: str = None,
        minimize: bool = None,
        topk: int = 1,
        resume: Union[str, Dict[str, str]] = None,
        mode: str = "model",
    ):
        """Init."""
        super().__init__(order=CallbackOrder.external)
        assert topk >= 1
        assert mode in (
            "model",
            "runner",
        ), "`CheckpointCallback` could work only in `model` or `runner` modes."
        assert mode == "model", "TODO"

        if minimize is not None:
            assert metric_key is not None, "please define the metric to track"
            self._minimize = minimize
            self.on_epoch_end = self.on_epoch_end_best
        else:
            self._minimize = False
            self.on_epoch_end = self.on_epoch_end_last

        # checkpointer info
        self.logdir = logdir
        self.mode = mode
        self.resume = resume

        # model selection info
        self.loader_key = loader_key
        self.metric_key = metric_key
        self.topk = topk
        self._storage: List[Checkpoint] = []
        os.makedirs(self.logdir, exist_ok=True)

    def save(self, runner: "IRunner", obj: Any, logprefix: str) -> str:
        logpath = f"{logprefix}.pth"
        if isinstance(obj, torch.nn.Module):
            runner.engine.wait_for_everyone()
            obj = runner.engine.unwrap_model(obj)
            runner.engine.save(obj.state_dict(), logpath)
        else:
            runner.engine.save(obj, logpath)
        return logpath

    def _handle_epoch(self, runner: "IRunner", score: float):
        obj = runner.model
        logprefix = f"{self.logdir}/{self.mode}.{runner.epoch_step:03d}"
        logpath = self.save(runner, obj, logprefix)
        self._storage.append(Checkpoint(obj=obj, logpath=logpath, metric=score))
        self._storage = sorted(
            self._storage, key=lambda x: x.metric, reverse=not self._minimize
        )
        if len(self._storage) > self.topk:
            last_item = self._storage.pop(-1)
            if os.path.isfile(last_item.logpath):
                try:
                    os.remove(last_item.logpath)
                except OSError:
                    pass
            elif os.path.isdir(last_item.logpath):
                shutil.rmtree(last_item.logpath, ignore_errors=True)
        with open(f"{self.logdir}/{self.mode}.storage.json", "w") as fout:
            stats = {
                "logdir": self.logdir,
                "topk": self.topk,
                "loader_key": self.loader_key,
                "metric_key": self.metric_key,
                "minimize": self._minimize,
            }
            storage = [{"logpath": x.logpath, "metric": x.metric} for x in self._storage]
            stats["storage"] = storage
            json.dump(stats, fout, indent=2, ensure_ascii=False)

    def on_experiment_start(self, runner: "IRunner") -> None:
        """Event handler."""
        self._storage: List[Checkpoint] = []

    def on_epoch_end_best(self, runner: "IRunner") -> None:
        if self.loader_key is not None:
            score = runner.epoch_metrics[self.loader_key][self.metric_key]
        else:
            score = runner.epoch_metrics[self.metric_key]
        self._handle_epoch(runner=runner, score=score)

        best_logprefix = f"{self.logdir}/{self.mode}.best"
        self.save(runner, self._storage[0].obj, best_logprefix)

    def on_epoch_end_last(self, runner: "IRunner") -> None:
        self._handle_epoch(runner=runner, score=runner.epoch_step)

    def on_experiment_end(self, runner: "IRunner") -> None:
        """Event handler."""
        if runner.engine.process_index == 0:
            # let's log Top-N base metrics
            log_message = "Top best models:\n"
            log_message += "\n".join(
                [
                    f"{checkpoint.logpath}\t{checkpoint.metric:3.4f}"
                    for checkpoint in self._storage
                ]
            )
            print(log_message)


__all__ = ["ICheckpointCallback", "CheckpointCallback"]
