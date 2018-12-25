import os
import logging
from typing import List, Dict
from collections import defaultdict

import numpy as np
import torch
import torch.nn.functional as F
import cv2

from catalyst.data.functional import compute_mixup_lambda, mixup_torch

from catalyst.dl.callbacks.utils import to_batch_metrics, \
    get_optimizer_momentum, scheduler_step
from catalyst.dl.state import RunnerState

from catalyst.dl.fp16 import Fp16Wrap, copy_params, copy_grads
from catalyst.dl.utils import UtilsFactory

cv2.setNumThreads(1)
cv2.ocl.setUseOpenCL(False)


class Callback:
    """
    An abstract class that all callback(e.g., Logger) classes extends from.
    Must be extended before usage.

    usage example:

    mode start (train/infer/debug)
        epoch start (one epoch - one run of every loader)
            loader start
                batch start
                batch handler
                batch end
            loader end
        epoch end
    mode end
    """

    def on_train_start(self, state):
        pass

    def on_train_end(self, state):
        pass

    def on_infer_start(self, state):
        pass

    def on_infer_end(self, state):
        pass

    def on_epoch_start(self, state):
        pass

    def on_epoch_end(self, state):
        pass

    def on_loader_start(self, state):
        pass

    def on_loader_end(self, state):
        pass

    def on_batch_start(self, state):
        pass

    def on_batch_end(self, state):
        pass


class CallbackCompose:
    def __init__(self, callbacks: Dict[str, Callback]):
        self.callbacks = callbacks

    def on_train_start(self, state):
        for key, value in self.callbacks.items():
            value.on_train_start(state=state)

    def on_train_end(self, state):
        for key, value in self.callbacks.items():
            value.on_train_end(state=state)

    def on_infer_start(self, state):
        for key, value in self.callbacks.items():
            value.on_infer_start(state=state)

    def on_infer_end(self, state):
        for key, value in self.callbacks.items():
            value.on_infer_end(state=state)

    def on_epoch_start(self, state):
        for key, value in self.callbacks.items():
            value.on_epoch_start(state=state)

    def on_epoch_end(self, state):
        for key, value in self.callbacks.items():
            value.on_epoch_end(state=state)

    def on_loader_start(self, state):
        for key, value in self.callbacks.items():
            value.on_loader_start(state=state)

    def on_loader_end(self, state):
        for key, value in self.callbacks.items():
            value.on_loader_end(state=state)

    def on_batch_start(self, state):
        for key, value in self.callbacks.items():
            value.on_batch_start(state=state)

    def on_batch_end(self, state):
        for key, value in self.callbacks.items():
            value.on_batch_end(state=state)


class Logger(Callback):
    """
    Logger callback, translates state.*_metrics to console and text file
    """

    def __init__(self, logdir: str = None):
        """
        :param logdir: log directory to use for text logging
        """
        self.logger = None
        self._logdir = logdir

    @property
    def logdir(self):
        return self._logdir

    @logdir.setter
    def logdir(self, value):
        self._logdir = value
        os.makedirs(value, exist_ok=True)
        log_filepath = os.path.join(value, "logs.txt")
        self.logger = self._get_logger(log_filepath)

    @staticmethod
    def _get_logger(log_filepath):
        logger = logging.getLogger(log_filepath)
        logger.setLevel(logging.DEBUG)
        fh = logging.FileHandler(log_filepath)
        fh.setLevel(logging.INFO)
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)
        formatter = logging.Formatter("[%(asctime)s] %(message)s")
        fh.setFormatter(formatter)
        ch.setFormatter(formatter)
        # add the handlers to the logger
        logger.addHandler(fh)
        logger.addHandler(ch)
        return logger

    @staticmethod
    def _get_metrics_string(metrics):
        return " | ".join(
            "{}: {:.5f}".format(k, v) for k, v in metrics.items()
        )

    def on_train_begin(self, state):
        if self.logger is not None:
            self.logger.info(
                "Starting training with params:\n{}\n\n".format(state)
            )

    def on_epoch_end(self, state):
        if self.logger is not None:
            for k, v in state.epoch_metrics.items():
                self.logger.info(
                    f"{state.epoch} * Epoch ({k}) metrics: "
                    f"{self._get_metrics_string(v)}"
                )
            self.logger.info("\n")


class TensorboardLogger(Callback):
    """
    Logger callback, translates state.*_metrics to tensorboard
    """

    def __init__(
        self,
        logdir: str = None,
        metric_names: List[str] = None,
        log_on_batch_end=True,
        log_on_epoch_end=True
    ):
        """
        :param logdir: directory where logs will be created
        :param metric_names: List of metric names to log.
            If none - logs everything.
        :param log_on_batch_end: Logs per-batch value of metrics,
            prepends 'batch_' prefix to their names.
        :param log_on_epoch_end: Logs per-epoch metrics if set True.
        """
        self.logdir = logdir
        self.metrics_to_log = metric_names
        self.log_on_batch_end = log_on_batch_end
        self.log_on_epoch_end = log_on_epoch_end

        # You definitely should log something)
        assert self.log_on_batch_end or self.log_on_epoch_end
        self.loggers = dict()

    def on_loader_start(self, state):
        lm = state.loader_mode
        if lm not in self.loggers:
            self.loggers[lm] = UtilsFactory.create_tflogger(
                logdir=self.logdir, name=lm
            )

    def _log_metrics(
        self, metrics: Dict[str, float], step: int, mode: str, suffix=""
    ):
        if self.metrics_to_log is None:
            self.metrics_to_log = list(metrics.keys())

        for name in self.metrics_to_log:
            if name in metrics:
                self.loggers[mode].add_scalar(
                    f"{name}{suffix}", metrics[name], step
                )

    def on_batch_end(self, state: RunnerState):
        if self.log_on_batch_end:
            mode = state.loader_mode

            to_batch_metrics(state=state, metric_key="lr")
            to_batch_metrics(state=state, metric_key="momentum")
            to_batch_metrics(state=state, metric_key="loss")

            self._log_metrics(
                metrics=state.batch_metrics,
                step=state.step,
                mode=mode,
                suffix="/batch"
            )

    def on_loader_end(self, state: RunnerState):
        if self.log_on_epoch_end:
            mode = state.loader_mode
            self._log_metrics(
                metrics=state.epoch_metrics[mode],
                step=state.epoch,
                mode=mode,
                suffix="/epoch"
            )


class CheckpointCallback(Callback):
    """
    Checkpoint callback to save/restore your mode/criterion/optimizer/metrics.
    """

    def __init__(
        self, logdir: str = None, save_n_best: int = 5, resume: str = None
    ):
        """
        :param logdir: log directory to use for checkpoint saving
        :param save_n_best: number of best checkpoint to keep
        :param resume: path to checkpoint to load and initialize runner state
        """
        self.logdir = logdir
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
            is_best=is_best,
            suffix=suffix
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
        assert self.logdir is not None, \
            "Please, specify logdir for callback usage"
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
        grad_clip_params = grad_clip_params or {}
        self.grad_clip_fn = UtilsFactory.create_grad_clip_fn(
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


class ClassificationLossCallback(Callback):
    def __init__(self, input_key: str = "targets", output_key: str = "logits"):
        self.input_key = input_key
        self.output_key = output_key

    def on_batch_end(self, state):
        state.loss = state.criterion(
            state.output[self.output_key], state.input[self.input_key]
        )


class InferCallback(Callback):
    def __init__(self, out_prefix=None):
        self.out_prefix = out_prefix
        self.predictions = defaultdict(lambda: [])

    def on_loader_start(self, state):
        self.predictions = defaultdict(lambda: [])

    def on_batch_end(self, state):
        dct = state.output
        dct = {key: value.detach().cpu().numpy() for key, value in dct.items()}
        for key, value in dct.items():
            self.predictions[key].append(value)

    def on_loader_end(self, state):
        self.predictions = {
            key: np.concatenate(value, axis=0)
            for key, value in self.predictions.items()
        }
        if self.out_prefix is not None:
            for key, value in self.predictions.items():
                np.save(
                    self.out_prefix.format(
                        suffix=".".join([state.loader_mode, key])
                    ), value
                )


class InferMaskCallback(Callback):
    def __init__(
        self,
        out_prefix=None,
        mean=None,
        std=None,
        mask_type="soft",
        threshold=None,
        input_key=None,
        output_key=None
    ):
        self.out_prefix = out_prefix
        self.predictions = defaultdict(lambda: [])
        self.mean = mean or np.array([0.485, 0.456, 0.406])
        self.std = std or np.array([0.229, 0.224, 0.225])
        assert mask_type in ["soft", "hard"], mask_type
        self.mask_type = mask_type
        self.threshold = threshold
        assert input_key is not None
        assert output_key is not None
        self.input_key = input_key
        self.output_key = output_key
        self.counter = 0

    def on_loader_start(self, state):
        lm = state.loader_mode
        os.makedirs(f"{self.out_prefix}/{lm}/", exist_ok=True)

    def on_batch_end(self, state):
        lm = state.loader_mode
        features = state.input[self.input_key]
        logits = state.output[self.output_key]
        logits = torch.unsqueeze_(logits, dim=1) \
            if len(logits.shape) < 4 \
            else logits

        if self.mask_type == "soft":
            probs = F.sigmoid(logits)
        else:
            probs = F.softmax(logits, dim=1)

        features = features.detach().cpu().numpy()
        features = np.transpose(features, (0, 2, 3, 1))

        probs = probs.detach().cpu().numpy()
        probs = np.transpose(probs, (0, 2, 3, 1))

        for i in range(probs.shape[0]):
            img = np.uint8(255 * (self.std * features[i] + self.mean))
            for t in range(probs.shape[-1]):
                mask = probs[i, :, :, t] > self.threshold \
                    if self.threshold is not None \
                    else probs[i, :, :, t]
                mask = np.float32(np.expand_dims(mask, -1))

                masked_img = img * mask

                # @TODO: better naming
                filename = f"{self.out_prefix}/{lm}/{self.counter}_{t}.jpg"
                cv2.imwrite(filename, masked_img)
            self.counter += 1


class MixupCallback(Callback):
    def __init__(
        self, mixup_keys: List[str], alpha: float, share_lambda: bool = True
    ):
        self.mixup_keys = mixup_keys
        self.alpha = alpha
        self.share_lambda = share_lambda

    def on_batch_start(self, state):
        lambda_ = compute_mixup_lambda(
            state.batch_size, self.alpha, self.share_lambda
        )
        for key in self.mixup_keys:
            state.input[key] = mixup_torch(state.input[key], lambda_=lambda_)
