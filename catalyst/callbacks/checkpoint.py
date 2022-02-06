from typing import Any, List
from collections import namedtuple
import json
import os
import shutil

import torch

from catalyst.core.callback import ICheckpointCallback
from catalyst.core.runner import IRunner
from catalyst.utils import (
    load_checkpoint,
    pack_checkpoint,
    save_checkpoint,
    unpack_checkpoint,
)

Checkpoint = namedtuple("Checkpoint", field_names=["obj", "logpath", "metric"])


class CheckpointCallback(ICheckpointCallback):
    """Checkpoint callback to save/restore your model/runner.

    Args:
        logdir: directory to store checkpoints
        loader_key: loader key for best model selection
            (based on metric score over the dataset)
        metric_key: metric key for best model selection
            (based on metric score over the dataset)
        minimize: boolean flag to minimize the required metric
        topk: number of best checkpoint to keep
        mode: checkpoint type to save, ``model`` or ``runner``. (default: model)
        save_last: boolean flag to save extra last checkpoint as ``{mode}.last.pth``
        save_best: boolean flag to save extra best checkpoint as ``{mode}.best.pth``
        resume_model: path to model checkpoint to load on experiment start
        resume_runner: path to runner checkpoint to load on experiment start
        load_best_on_end: boolean flag to load best model on experiment end
    """

    def __init__(
        self,
        logdir: str,
        loader_key: str = None,
        metric_key: str = None,
        minimize: bool = None,
        topk: int = 1,
        mode: str = "model",
        save_last: bool = True,
        save_best: bool = True,
        resume_model: str = None,
        resume_runner: str = None,
        load_best_on_end: bool = False,
    ):
        """Init."""
        super().__init__()
        assert topk >= 1
        assert mode in (
            "model",
            "runner",
        ), "`CheckpointCallback` could work only in `model` or `runner` modes."

        if minimize is not None:
            assert metric_key is not None, "please define the metric to track"
            self._minimize = minimize
            self.on_epoch_end = self.on_epoch_end_best
        else:
            self._minimize = False
            self.on_epoch_end = self.on_epoch_end_last

        self.logdir = logdir
        self.loader_key = loader_key
        self.metric_key = metric_key
        self.topk = topk
        self._storage: List[Checkpoint] = []
        self.save_last = save_last
        self.save_best = save_best
        self.mode = mode
        self._resume_model = resume_model
        self._resume_runner = resume_runner
        self.load_best_on_end = load_best_on_end
        os.makedirs(self.logdir, exist_ok=True)

    def _save(self, runner: "IRunner", obj: Any, logprefix: str) -> str:
        logpath = f"{logprefix}.pth"
        if self.mode == "model":
            if issubclass(obj.__class__, torch.nn.Module):
                runner.engine.wait_for_everyone()
                obj = runner.engine.unwrap_model(obj)
                runner.engine.save(obj.state_dict(), logpath)
            elif isinstance(obj, dict):
                # obj = dict(model=obj)  # noqa: C408
                checkpoint = pack_checkpoint(model=obj)
                save_checkpoint(checkpoint, logpath)
            else:
                raise NotImplementedError()
        else:
            checkpoint = pack_checkpoint(**obj)
            save_checkpoint(checkpoint, logpath)
        return logpath

    def _load(
        self,
        runner: "IRunner",
        resume_logpath: Any = None,
        resume_model: str = None,
        resume_runner: str = None,
    ):
        if resume_logpath is not None:
            runner.engine.wait_for_everyone()
            if self.mode == "model":
                try:
                    unwrapped_model = runner.engine.unwrap_model(runner.model)
                    unwrapped_model.load_state_dict(load_checkpoint(resume_logpath))
                except BaseException:
                    checkpoint = load_checkpoint(resume_logpath)
                    unpack_checkpoint(checkpoint=checkpoint, model=runner.model)
            else:
                checkpoint = load_checkpoint(resume_logpath)
                unpack_checkpoint(checkpoint=checkpoint, model=runner.model)
        if resume_runner is not None:
            runner.engine.wait_for_everyone()
            checkpoint = load_checkpoint(resume_runner)
            unpack_checkpoint(
                checkpoint=checkpoint,
                model=runner.model,
                criterion=runner.criterion,
                optimizer=runner.optimizer,
                scheduler=runner.scheduler,
            )
            runner.epoch_step = checkpoint["epoch_step"]
            runner.batch_step = checkpoint["batch_step"]
            runner.sample_step = checkpoint["sample_step"]
        if resume_model is not None:
            runner.engine.wait_for_everyone()
            unwrapped_model = runner.engine.unwrap_model(runner.model)
            unwrapped_model.load_state_dict(load_checkpoint(resume_model))
        # if resume_runner is not None or resume_model is not None:
        #     runner.model, runner.optimizer = runner.engine.prepare(
        #         runner.model, runner.optimizer
        #     )

    def _handle_epoch(self, runner: "IRunner", score: float):
        if self.mode == "model":
            obj = runner.model
        else:
            obj = dict(  # noqa: C408
                model=runner.model,
                criterion=runner.criterion,
                optimizer=runner.optimizer,
                scheduler=runner.scheduler,
                epoch_step=runner.epoch_step,
                batch_step=runner.batch_step,
                sample_step=runner.sample_step,
            )
        if self.save_last:
            # @TODO: simplify it
            logprefix = f"{self.logdir}/{self.mode}.last"
            logpath = self._save(runner, obj, logprefix)

        logprefix = f"{self.logdir}/{self.mode}.{runner.epoch_step:04d}"
        logpath = self._save(runner, obj, logprefix)
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
                "logdir": str(self.logdir),
                "topk": self.topk,
                "loader_key": self.loader_key,
                "metric_key": self.metric_key,
                "minimize": self._minimize,
            }
            storage = [
                {"logpath": str(x.logpath), "metric": x.metric} for x in self._storage
            ]
            stats["storage"] = storage
            json.dump(stats, fout, indent=2, ensure_ascii=False)

    def on_experiment_start(self, runner: "IRunner") -> None:
        """Event handler."""
        self._storage: List[Checkpoint] = []
        # assert issubclass(runner.model.__class__, torch.nn.Module), (
        #     "Could not understand the model class. "
        #     "Do you mean ``nn.Module`` or ``nn.ModuleDict``?"
        # )
        self._load(
            runner=runner,
            resume_runner=self._resume_runner,
            resume_model=self._resume_model,
        )

    def on_epoch_end_best(self, runner: "IRunner") -> None:
        """Event handler."""
        if self.loader_key is not None:
            score = runner.epoch_metrics[self.loader_key][self.metric_key]
        else:
            score = runner.epoch_metrics[self.metric_key]
        self._handle_epoch(runner=runner, score=score)

        if self.save_best:
            best_logprefix = f"{self.logdir}/{self.mode}.best"
            self._save(runner, self._storage[0].obj, best_logprefix)

    def on_epoch_end_last(self, runner: "IRunner") -> None:
        """Event handler."""
        self._handle_epoch(runner=runner, score=runner.epoch_step)

    def on_experiment_end(self, runner: "IRunner") -> None:
        """Event handler."""
        if runner.engine.process_index == 0:
            log_message = "Top models:\n"
            log_message += "\n".join(
                [
                    f"{checkpoint.logpath}\t{checkpoint.metric:3.4f}"
                    for checkpoint in self._storage
                ]
            )
            print(log_message)
        if self.load_best_on_end:
            self._load(runner=runner, resume_logpath=self._storage[0].logpath)


__all__ = ["CheckpointCallback"]
