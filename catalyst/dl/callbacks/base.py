import os
import safitty
import torch
from typing import Dict

from catalyst.contrib.scheduler import OneCycleLR, BatchScheduler
from catalyst.dl.state import RunnerState
from catalyst.dl.utils import UtilsFactory, get_optimizer_momentum
from catalyst.rl.registry import GRAD_CLIPPERS
from .core import Callback


class CheckpointCallback(Callback):
    """
    Checkpoint callback to save/restore your model/criterion/optimizer/metrics.
    """

    def __init__(
        self, save_n_best: int = 3, resume: str = None, resume_dir: str = None
    ):
        """
        Args:
            save_n_best: number of best checkpoint to keep
            resume: path to checkpoint to load and initialize runner state
        """
        self.save_n_best = save_n_best
        self.resume = resume
        self.resume_dir = resume_dir
        self.top_best_metrics = []

        self._keys_from_state = ["resume", "resume_dir"]

    @staticmethod
    def load_checkpoint(*, filename, state):
        if os.path.isfile(filename):
            print("=> loading checkpoint \"{}\"".format(filename))
            checkpoint = UtilsFactory.load_checkpoint(filename)

            state.epoch = checkpoint["epoch"]

            UtilsFactory.unpack_checkpoint(
                checkpoint,
                model=state.model,
                criterion=state.criterion,
                optimizer=state.optimizer,
                scheduler=state.scheduler
            )

            print(
                "loaded checkpoint \"{}\" (epoch {})".format(
                    filename, checkpoint["epoch"]
                )
            )
        else:
            raise Exception("no checkpoint found at \"{}\"".format(filename))

    def save_checkpoint(
        self,
        logdir,
        checkpoint,
        is_best,
        save_n_best=5,
        main_metric="loss",
        minimize_metric=True
    ):
        suffix = f"{checkpoint['stage']}.{checkpoint['epoch']}"
        filepath = UtilsFactory.save_checkpoint(
            logdir=f"{logdir}/checkpoints/",
            checkpoint=checkpoint,
            suffix=suffix,
            is_best=is_best,
            is_last=True
        )

        checkpoint_metric = checkpoint["valid_metrics"][main_metric]
        self.top_best_metrics.append((filepath, checkpoint_metric))
        self.top_best_metrics = sorted(
            self.top_best_metrics,
            key=lambda x: x[1],
            reverse=not minimize_metric
        )
        if len(self.top_best_metrics) > save_n_best:
            last_item = self.top_best_metrics.pop(-1)
            last_filepath = last_item[0]
            os.remove(last_filepath)

    def pack_checkpoint(self, **kwargs):
        return UtilsFactory.pack_checkpoint(**kwargs)

    def on_stage_start(self, state):
        for key in self._keys_from_state:
            value = getattr(state, key, None)
            if value is not None:
                setattr(self, key, value)

        if self.resume_dir is not None:
            self.resume = str(self.resume_dir) + "/" + str(self.resume)

        if self.resume is not None:
            self.load_checkpoint(filename=self.resume, state=state)

    def on_epoch_end(self, state: RunnerState):
        if state.stage.startswith("infer"):
            return

        checkpoint = self.pack_checkpoint(
            model=state.model,
            criterion=state.criterion,
            optimizer=state.optimizer,
            scheduler=state.scheduler,
            epoch_metrics=dict(state.metrics.epoch_values),
            valid_metrics=dict(state.metrics.valid_values),
            stage=state.stage,
            epoch=state.epoch,
            checkpoint_data=state.checkpoint_data
        )
        self.save_checkpoint(
            logdir=state.logdir,
            checkpoint=checkpoint,
            is_best=state.metrics.is_best,
            save_n_best=self.save_n_best,
            main_metric=state.main_metric,
            minimize_metric=state.minimize_metric
        )

    def on_stage_end(self, state):
        print("Top best models:")
        top_best_metrics_str = "\n".join(
            [
                "{filepath}\t{metric:3.4f}".format(
                    filepath=filepath, metric=metric
                ) for filepath, metric in self.top_best_metrics
            ]
        )
        print(top_best_metrics_str)


class IterationCheckpointCallback(Callback):
    """
    Iteration checkpoint callback to save your model/criterion/optimizer
    """

    def __init__(
        self,
        save_n_last: int = 3,
        num_iters: int = 100,
        stage_restart: bool = True
    ):

        """
        :param save_n_last: number of last checkpoint to keep
        :param num_iters: save the checkpoint every `num_iters`
        :param stage_restart: restart counter every stage or not
        """
        self.save_n_last = save_n_last
        self.num_iters = num_iters
        self.stage_restart = stage_restart
        self._iteration_counter = 0
        self.last_checkpoints = []

    def save_checkpoint(
        self,
        logdir,
        checkpoint,
        save_n_last
    ):
        suffix = f"{checkpoint['stage']}." \
                 f"epoch.{checkpoint['epoch']}." \
                 f"iter.{self._iteration_counter}"

        filepath = UtilsFactory.save_checkpoint(
            logdir=f"{logdir}/checkpoints/",
            checkpoint=checkpoint,
            suffix=suffix,
            is_best=False,
            is_last=False
        )

        self.last_checkpoints.append(filepath)
        if len(self.last_checkpoints) > save_n_last:
            top_filepath = self.last_checkpoints.pop(0)
            os.remove(top_filepath)

        print(f"\nSaved checkpoint at {filepath}")

    def pack_checkpoint(self, **kwargs):
        return UtilsFactory.pack_checkpoint(**kwargs)

    def on_stage_start(self, state):
        if self.stage_restart:
            self._iteration_counter = 0

    def on_batch_end(self, state):
        self._iteration_counter += 1
        if self._iteration_counter % self.num_iters == 0:
            checkpoint = self.pack_checkpoint(
                model=state.model,
                criterion=state.criterion,
                optimizer=state.optimizer,
                scheduler=state.scheduler,
                epoch_metrics=None,
                valid_metrics=None,
                stage=state.stage,
                epoch=state.epoch
            )
            self.save_checkpoint(
                logdir=state.logdir,
                checkpoint=checkpoint,
                save_n_last=self.save_n_last
            )


class OptimizerCallback(Callback):
    """
    Optimizer callback, abstraction over optimizer step.
    """

    def __init__(
        self,
        grad_clip_params: Dict = None,
        accumulation_steps: int = 1,
        optimizer_key: str = None,
        loss_key: str = None,
        prefix: str = None
    ):
        """
        @TODO: docs
        """

        grad_clip_params = grad_clip_params or {}
        self.grad_clip_fn = GRAD_CLIPPERS.get_from_params(**grad_clip_params)

        self.accumulation_steps = accumulation_steps
        self.optimizer_key = optimizer_key
        self.loss_key = loss_key
        self.prefix = prefix
        self._optimizer_wd = 0
        self._accumulation_counter = 0

    @staticmethod
    def grad_step(*, optimizer, optimizer_wd=0, grad_clip_fn=None):
        for group in optimizer.param_groups:
            if optimizer_wd > 0:
                for param in group["params"]:
                    param.data = param.data.add(
                        -optimizer_wd * group["lr"], param.data
                    )
            if grad_clip_fn is not None:
                grad_clip_fn(group["params"])
        optimizer.step()

    def on_stage_start(self, state: RunnerState):
        optimizer = state.get_key(
            key="optimizer", inner_key=self.optimizer_key
        )
        assert optimizer is not None
        lr = optimizer.defaults["lr"]
        momentum = get_optimizer_momentum(optimizer)
        state.set_key(lr, "lr", inner_key=self.optimizer_key)
        state.set_key(momentum, "momentum", inner_key=self.optimizer_key)

    def on_epoch_start(self, state):
        optimizer = state.get_key(
            key="optimizer", inner_key=self.optimizer_key
        )
        self._optimizer_wd = optimizer.param_groups[0].get("weight_decay", 0.0)
        optimizer.param_groups[0]["weight_decay"] = 0.0

    def on_batch_start(self, state):
        state.loss = None

    def on_batch_end(self, state):
        loss = state.get_key(key="loss", inner_key=self.loss_key)
        if isinstance(loss, dict):
            loss = list(loss.values())
        if isinstance(loss, list):
            loss = torch.mean(torch.stack(loss))

        if self.prefix is not None:
            state.metrics.add_batch_value(metrics_dict={
                self.prefix: loss.item(),
            })

        if not state.need_backward:
            return

        self._accumulation_counter += 1
        model = state.model
        optimizer = state.get_key(
            key="optimizer", inner_key=self.optimizer_key
        )

        # This is very hacky check whether we have AMP optimizer and this may
        # change in future.
        # But alternative solution is to have AmpOptimizerCallback.
        # or expose another c'tor argument.
        if hasattr(optimizer, "_amp_stash"):
            from apex import amp
            with amp.scale_loss(loss, optimizer) as scaled_loss:
                scaled_loss.backward()
        else:
            loss.backward()

        if self._accumulation_counter % self.accumulation_steps == 0:
            self.grad_step(
                optimizer=optimizer,
                optimizer_wd=self._optimizer_wd,
                grad_clip_fn=self.grad_clip_fn
            )
            model.zero_grad()
            self._accumulation_counter = 0

    def on_epoch_end(self, state):
        optimizer = state.get_key(
            key="optimizer", inner_key=self.optimizer_key
        )
        optimizer.param_groups[0]["weight_decay"] = self._optimizer_wd


class SchedulerCallback(Callback):
    def __init__(
        self,
        scheduler_key: str = None,
        mode: str = None,
        reduce_metric: str = "loss"
    ):
        self.scheduler_key = scheduler_key
        self.mode = mode
        self.reduce_metric = reduce_metric

    def step(self, state: RunnerState):
        scheduler = state.get_key(
            key="scheduler", inner_key=self.scheduler_key
        )

        valid_metric = \
            safitty.get(state.metrics.valid_values, self.reduce_metric)
        lr, momentum = self._scheduler_step(
            scheduler=scheduler,
            valid_metric=valid_metric
        )

        state.set_key(lr, key="lr", inner_key=self.scheduler_key)
        state.set_key(momentum, key="momentum", inner_key=self.scheduler_key)

    def on_stage_start(self, state: RunnerState):
        scheduler = state.get_key(
            key="scheduler", inner_key=self.scheduler_key
        )
        assert scheduler is not None

        if self.mode is None:
            if isinstance(scheduler, BatchScheduler):
                self.mode = "batch"
            else:
                self.mode = "epoch"

        if isinstance(scheduler, OneCycleLR) and self.mode == "batch":
            scheduler.reset()

    def on_loader_start(self, state: RunnerState):
        scheduler = state.get_key(
            key="scheduler", inner_key=self.scheduler_key
        )
        if state.loader_name.startswith("train") and \
                isinstance(scheduler, OneCycleLR) and self.mode == "batch":
            scheduler.recalculate(
                loader_len=state.loader_len,
                current_step=state.stage_epoch
            )

    def on_batch_end(self, state):
        if self.mode == "batch":
            self.step(state=state)

    def on_epoch_end(self, state):
        if self.mode == "epoch":
            self.step(state=state)

    @staticmethod
    def _scheduler_step(
            scheduler,
            valid_metric=None,
    ):
        if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
            scheduler.step(valid_metric)
            lr = safitty.get(scheduler.optimizer.param_groups, 0, "lr")
        else:
            scheduler.step()
            lr = scheduler.get_lr()[0]

        momentum = get_optimizer_momentum(scheduler.optimizer)

        return lr, momentum


class LossCallback(Callback):
    def __init__(
        self,
        input_key: str = "targets",
        output_key: str = "logits",
        prefix: str = "loss",
        criterion_key: str = None,
        loss_key: str = None,
        multiplier: float = 1.0
    ):
        self.input_key = input_key
        self.output_key = output_key
        self.prefix = prefix
        self.criterion_key = criterion_key
        self.loss_key = loss_key
        self.multiplier = multiplier

    def _add_loss_to_state(self, state, loss):
        if self.loss_key is None:
            if state.loss is not None:
                if isinstance(state.loss, list):
                    state.loss.append(loss)
                else:
                    state.loss = [state.loss, loss]
            else:
                state.loss = loss
        else:
            if state.loss is not None:
                assert isinstance(state.loss, dict)
                state.loss[self.loss_key] = loss
            else:
                state.loss = {self.loss_key: loss}

    def _compute_loss(self, state, criterion):
        loss = criterion(
            state.output[self.output_key],
            state.input[self.input_key]
        )
        return loss

    def on_stage_start(self, state):
        assert state.criterion is not None

    def on_batch_end(self, state):
        criterion = state.get_key(
            key="criterion", inner_key=self.criterion_key
        )

        loss = self._compute_loss(state, criterion) * self.multiplier

        state.metrics.add_batch_value(metrics_dict={
            self.prefix: loss.item(),
        })

        self._add_loss_to_state(state, loss)


class EarlyStoppingCallback(Callback):
    def __init__(
        self,
        patience: int,
        metric: str = "loss",
        minimize: bool = True,
        min_delta: float = 1e-6
    ):
        self.best_score = None
        self.metric = metric
        self.patience = patience
        self.num_bad_epochs = 0
        self.is_better = None

        if minimize:
            self.is_better = lambda score, best: score <= (best - min_delta)
        else:
            self.is_better = lambda score, best: score >= (best - min_delta)

    def on_epoch_end(self, state: RunnerState) -> None:
        if state.stage.startswith("infer"):
            return

        score = state.metrics.valid_values[self.metric]
        if self.best_score is None:
            self.best_score = score
        if self.is_better(score, self.best_score):
            self.num_bad_epochs = 0
            self.best_score = score
        else:
            self.num_bad_epochs += 1

        if self.num_bad_epochs >= self.patience:
            print(f"Early stop at {state.stage_epoch} epoch")
            state.early_stop = True
