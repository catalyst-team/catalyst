import os
from typing import Dict
import torch
from catalyst.dl.utils import UtilsFactory
from .core import Callback
from catalyst.dl.fp16 import Fp16Wrap, copy_params, copy_grads
from .utils import get_optimizer_momentum, scheduler_step


class LoggerCallback(Callback):
    """
    Base class for anything that needs logdir to be specified in 'train' mode.
    """

    def __init__(self, logdir: str = None):
        """
        Args:
            logdir: directory where logs will be created
                If directory doesn't exists it will be created
                If None, RunnerState.logdir will be used
        """
        self.logdir = logdir

    def on_train_start(self, state):
        assert self.logdir or state.logdir, \
            "Please, specify logdir for callback usage"
        if self.logdir is None:
            self.logdir = state.logdir
        os.makedirs(self.logdir, exist_ok=True)


class CheckpointCallback(LoggerCallback):
    """
    Checkpoint callback to save/restore your model/criterion/optimizer/metrics.
    """

    def __init__(
        self, logdir: str = None, save_n_best: int = 5, resume: str = None
    ):
        """
        :param logdir: log directory to use for checkpoint saving
        :param save_n_best: number of best checkpoint to keep
        :param resume: path to checkpoint to load and initialize runner state
        """
        super().__init__(logdir)
        self.save_n_best = save_n_best
        self.resume = resume
        self.top_best_metrics = []

    @staticmethod
    def load_checkpoint(*, filename, state):
        if os.path.isfile(filename):
            print("=> loading checkpoint \"{}\"".format(filename))
            checkpoint = UtilsFactory.load_checkpoint(filename)

            state.epoch = checkpoint["epoch"]
            state.best_metrics = checkpoint["best_metrics"]

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
            logdir=logdir,
            checkpoint=checkpoint,
            suffix=suffix,
            is_best=is_best,
            is_last=True
        )
        checkpoint_metric = checkpoint["valid_metrics"].get(main_metric, None)
        checkpoint_metric = checkpoint_metric or checkpoint.get("epoch", -1)
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

    def on_mode_start(self, state):
        if self.resume is not None:
            self.load_checkpoint(filename=self.resume, state=state)

    def on_train_start(self, state):
        super().on_train_start(state)
        return self.on_mode_start(state=state)

    def on_infer_start(self, state):
        return self.on_mode_start(state=state)

    def on_epoch_end(self, state):
        if state.mode == "infer":
            return

        checkpoint = self.pack_checkpoint(
            model=state.model,
            criterion=state.criterion,
            optimizer=state.optimizer,
            scheduler=state.scheduler,
            valid_metrics=dict(state.valid_metrics),  # @TODO: save defaultdict
            epoch_metrics=dict(state.epoch_metrics),  # @TODO: save defaultdict
            best_metrics=dict(state.best_metrics),  # @TODO: save defaultdict
            stage=state.stage,
            epoch=state.epoch
        )
        self.save_checkpoint(
            logdir=self.logdir,
            checkpoint=checkpoint,
            is_best=state.is_best_epoch,
            save_n_best=self.save_n_best,
            main_metric=state.main_metric,
            minimize_metric=state.minimize_metric
        )

    def on_train_end(self, state):
        print("Top best models:")
        top_best_metrics_str = "\n".join(
            [
                "{filepath}\t{metric:.4f}".format(
                    filepath=filepath, metric=metric
                ) for filepath, metric in self.top_best_metrics
            ]
        )
        print(top_best_metrics_str)


class OptimizerCallback(Callback):
    """
    Optimizer callback, abstraction over optimizer step.
    """

    def __init__(
        self,
        grad_clip_params: Dict = None,
        fp16_grad_scale: float = 128.0,
        accumulation_steps: int = 1,
        optimizer_key: str = None,
        loss_key: str = None
    ):
        """
        @TODO: docs
        """
        # hack to prevent cycle imports
        from catalyst.contrib.registry import Registry

        grad_clip_params = grad_clip_params or {}
        self.grad_clip_fn = Registry.get_grad_clip_fn(
            **grad_clip_params
        )
        self.fp16 = False
        self.fp16_grad_scale = fp16_grad_scale
        self.accumulation_steps = accumulation_steps
        self.optimizer_key = optimizer_key
        self.loss_key = loss_key
        self.optimizer_wd = 0
        self.accumulation_counter = 0

    def on_train_start(self, state):
        self.fp16 = isinstance(state.model, Fp16Wrap)
        optimizer = state.get_key(
            key="optimizer", inner_key=self.optimizer_key
        )
        lr = optimizer.defaults["lr"]
        momentum = get_optimizer_momentum(optimizer)
        state.set_key(lr, "lr", inner_key=self.optimizer_key)
        state.set_key(momentum, "momentum", inner_key=self.optimizer_key)

    def on_epoch_start(self, state):
        optimizer = state.get_key(
            key="optimizer", inner_key=self.optimizer_key
        )
        self.optimizer_wd = optimizer.param_groups[0].get("weight_decay", 0.0)
        optimizer.param_groups[0]["weight_decay"] = 0.0

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

    def on_batch_end(self, state):
        if not state.is_train:
            return

        self.accumulation_counter += 1
        if not self.fp16:
            model = state.model
            optimizer = state.get_key(
                key="optimizer", inner_key=self.optimizer_key
            )
            loss = state.get_key(key="loss", inner_key=self.loss_key)
            loss.backward()

            if (self.accumulation_counter + 1) % self.accumulation_steps == 0:
                self.grad_step(
                    optimizer=optimizer,
                    optimizer_wd=self.optimizer_wd,
                    grad_clip_fn=self.grad_clip_fn
                )
                model.zero_grad()
                self.accumulation_counter = 0
        else:
            model = state.model
            model.zero_grad()
            optimizer = state.get_key(
                key="optimizer", inner_key=self.optimizer_key
            )
            loss = state.get_key(key="loss", inner_key=self.optimizer_key)
            scaled_loss = self.fp16_grad_scale * loss.float()
            scaled_loss.backward()

            master_params = list(optimizer.param_groups[0]["params"])
            model_params = list(
                filter(lambda p: p.requires_grad, model.parameters())
            )
            copy_grads(source=model_params, target=master_params)
            for param in master_params:
                param.grad.data.mul_(1. / self.fp16_grad_scale)
            self.grad_step(
                optimizer=optimizer,
                optimizer_wd=self.optimizer_wd,
                grad_clip_fn=self.grad_clip_fn
            )
            copy_params(source=master_params, target=model_params)
            torch.cuda.synchronize()

    def on_epoch_end(self, state):
        optimizer = state.get_key(
            key="optimizer", inner_key=self.optimizer_key
        )
        optimizer.param_groups[0]["weight_decay"] = self.optimizer_wd


class SchedulerCallback(Callback):
    def __init__(
        self,
        scheduler_key: str = None,
        mode: str = "epoch",
        reduce_metric: str = None
    ):
        self.scheduler_key = scheduler_key
        self.mode = mode
        self.reduce_metric = reduce_metric

    def step(self, state):
        scheduler = state.get_key(
            key="scheduler", inner_key=self.scheduler_key
        )

        lr, momentum = scheduler_step(
            scheduler=scheduler,
            valid_metric=state.valid_metrics[self.reduce_metric]
        )

        state.set_key(lr, key="lr", inner_key=self.scheduler_key)
        state.set_key(momentum, key="momentum", inner_key=self.scheduler_key)

    def on_batch_end(self, state):
        if self.mode == "batch":
            self.step(state=state)

    def on_epoch_end(self, state):
        if self.mode == "epoch":
            self.step(state=state)


class LossCallback(Callback):
    def __init__(self, input_key: str = "targets", output_key: str = "logits"):
        self.input_key = input_key
        self.output_key = output_key

    def on_batch_end(self, state):
        state.loss = state.criterion(
            state.output[self.output_key], state.input[self.input_key]
        )
