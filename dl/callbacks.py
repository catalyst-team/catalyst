import os
import time
import logging
import numpy as np
from collections import defaultdict
from typing import Tuple, List
import torch

from catalyst.data.functional import compute_mixup_lambda, mixup_torch
from catalyst.dl.callback import Callback
from catalyst.utils.metrics import precision, get_iou_vector, mapk, dice_accuracy, F_score, mae
from catalyst.utils.fp16 import Fp16Wrap, copy_params, copy_grads
from catalyst.utils.factory import UtilsFactory


class BaseMetrics(Callback):
    def __init__(self):
        self.time = time.time()

    def on_loader_start(self, state):
        self.time = time.time()

    def on_batch_start(self, state):
        state.batch_metrics["data time"] = time.time() - self.time

    def on_batch_end(self, state):
        bs = state.batch_size
        elapsed_time = time.time() - self.time

        state.batch_metrics["batch time"] = elapsed_time
        state.batch_metrics["sample per second"] = bs / elapsed_time

        for key, value in state.lr.items():
            state.batch_metrics[f"lr_{key}"] = value

        for key, value in state.momentum.items():
            state.batch_metrics[f"momentum_{key}"] = value

        for key, value in state.loss.items():
            state.batch_metrics[f"loss_{key}"] = value.item()

        self.time = time.time()


class PrecisionCallback(Callback):
    """
    Precision metric callback.
    """

    def __init__(
            self,
            input_key: str = "targets",
            output_key: str = "logits",
            precision_args: List[int] = None,
            prefix="precision"):
        """
        :param input_key: input key to use for precision calculation;
            specifies our `y_true`.
        :param output_key: output key to use for precision calculation;
            specifies our `y_pred`.
        :param precision_args: specifies which precision@K to log.
            [1] - accuracy
            [1, 3] - accuracy and precision@3
            [1, 3, 5] - precision at 1, 3 and 5
        """
        self.input_key = input_key
        self.output_key = output_key
        self.precision_args = precision_args or [1, 3, 5]
        self.prefix = prefix

    def on_batch_end(self, state):
        prec = precision(
            state.output[self.output_key],
            state.input[self.input_key],
            topk=self.precision_args)
        for p, metric in zip(self.precision_args, prec):
            key = f"{self.prefix}{p:02}"
            metric_ = metric.item()
            state.batch_metrics[key] = metric_


class MapKCallback(Callback):
    """
    mAP@k metric callback.
    """

    def __init__(self,
                 input_key: str = "targets",
                 output_key: str = "logits",
                 map_args: List[int] = None):
        """
        :param input_key: input key to use for precision calculation;
            specifies our `y_true`.
        :param output_key: output key to use for precision calculation;
            specifies our `y_pred`.
        :param map_args: specifies which map@K to log.
            [1] - map@1
            [1, 3] - map@1 and map@3
            [1, 3, 5] - map@1, map@3 and map@5
        """
        super().__init__()
        self.input_key = input_key
        self.output_key = output_key
        self.map_args = map_args or [1, 3, 5]

    def on_batch_end(self, state):
        mapatk = mapk(
            state.output[self.output_key],
            state.input[self.input_key],
            topk=self.map_args)
        for p, metric in zip(self.map_args, mapatk):
            key = "map{:02}".format(p)
            metric_ = metric.item()
            state.batch_metrics[key] = metric_


class DiceCallback(Callback):
    """
    Dice metric callback.
    """

    def __init__(self,
                 input_key: str = "targets",
                 output_key: str = "logits",
                 threshold: float = None):
        """
        :param input_key: input key to use for precision calculation;
            specifies our `y_true`.
        :param output_key: output key to use for precision calculation;
            specifies our `y_pred`.
        :param threshold: threshold.
        """
        super().__init__()
        self.input_key = input_key
        self.output_key = output_key
        self.threshold = threshold

    def on_batch_end(self, state):
        dice = dice_accuracy(
            state.output[self.output_key],
            state.input[self.input_key],
            threshold=self.threshold
        )
        key = "dice"
        state.batch_metrics[key] = dice


class IOUCallback(Callback):
    """
    IOU metric callback.
    """

    def __init__(self,
                 input_key: str = "targets",
                 output_key: str = "logits"):
        """
        :param input_key: input key to use for precision calculation;
            specifies our `y_true`.
        :param output_key: output key to use for precision calculation;
            specifies our `y_pred`.
        :param precision_args: specifies which precision@K to log.
        """
        super().__init__()
        self.input_key = input_key
        self.output_key = output_key

    def on_batch_end(self, state):
        msk_vpreds = state.output[self.output_key]
        valid_msks = state.input[self.input_key]

        # msk_vpreds = msk_vpreds.sigmoid()
        msk_vpreds = msk_vpreds.detach().cpu().numpy()
        valid_msks = valid_msks.detach().cpu().numpy()

        iou = get_iou_vector(valid_msks, msk_vpreds)
        key = "iou"
        state.batch_metrics[key] = iou


class F2Callback(Callback):
    """
    F2 metric callback.
    """

    def __init__(self,
                 input_key: str = "targets",
                 output_key: str = "logits"):
        """
        :param input_key: input key to use for precision calculation;
            specifies our `y_true`.
        :param output_key: output key to use for precision calculation;
            specifies our `y_pred`.
        """
        super().__init__()
        self.input_key = input_key
        self.output_key = output_key

    def on_loader_start(self, state):
        self.outputs = torch.Tensor([])
        self.labels = torch.Tensor([])

    def on_batch_end(self, state):
        label = state.input[self.input_key].detach().cpu()
        output = state.output[self.output_key].detach().cpu()
        self.outputs = torch.cat([self.outputs, output], 0)
        self.labels = torch.cat([self.labels, label], 0)

    def on_loader_end(self, state):
        fscore = F_score(self.labels, self.outputs)
        key = "f2_score"
        state.epoch_metrics[state.loader_mode][key] = fscore
        print(f"{state.loader_mode} ", fscore)


class MAECallback(Callback):
    """
    MAE metric callback.
    """

    def __init__(self,
                 input_key: str = "targets",
                 output_key: str = "logits",
                 max_value : float = 90):
        """
        :param input_key: input key to use for precision calculation;
            specifies our `y_true`.
        :param output_key: output key to use for precision calculation;
            specifies our `y_pred`.
        """
        super().__init__()
        self.input_key = input_key
        self.output_key = output_key
        self.max_value = max_value

    def on_loader_start(self, state):
        self.outputs = torch.Tensor([])
        self.labels = torch.Tensor([])

    def on_batch_end(self, state):
        label = state.input[self.input_key].float().detach().cpu()
        output = state.output[self.output_key].float().detach().cpu()
        self.outputs = torch.cat([self.outputs, output], 0)
        self.labels = torch.cat([self.labels, label], 0)

    def on_loader_end(self, state):
        import pdb
        mae_score = mae(self.labels, self.outputs)
        # pdb.set_trace()
        mae_score =  mae_score * self.max_value
        key = "mae"
        state.epoch_metrics[state.loader_mode][key] = mae_score
        print(f"{state.loader_mode} ", mae_score)


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

    def on_train_begin(self, state):
        if self.logger is not None:
            self.logger.info(
                "Starting training with params:\n{}\n\n".format(state))

    def on_epoch_end(self, state):
        if self.logger is not None:
            for k, v in state.epoch_metrics.items():
                self.logger.info(
                    f"{state.epoch} * Epoch ({k}) metrics: " +
                    self._get_metrics_string(v))
            self.logger.info("\n")

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

    def _get_metrics_string(self, metrics):
        return " | ".join(
            "{}: {:.5f}".format(k, v) for k, v in metrics.items())


class TensorboardLogger(Callback):
    """
    Logger callback, translates state.*_metrics to tensorboard
    """

    def __init__(self, logdir: str = None):
        """
        :param logdir: log directory to use for tf logging
        """
        self.logdir = logdir
        self.loggers = dict()

    def on_loader_start(self, state):
        lm = state.loader_mode
        if lm not in self.loggers:
            self.loggers[lm] = UtilsFactory.create_tflogger(
                logdir=self.logdir, name=lm)

    def on_batch_end(self, state):
        lm = state.loader_mode
        for key, value in state.batch_metrics.items():
            self.loggers[lm].add_scalar(key, value, state.step)

    def on_loader_end(self, state):
        lm = state.loader_mode

        for key, value in state.epoch_metrics[lm].items():
            self.loggers[lm].add_scalar(f"epoch {key}", value, state.epoch)


class CheckpointCallback(Callback):
    """
    Checkpoint callback to save/restore your mode/criterion/optimizer/metrics.
    """

    def __init__(
            self,
            logdir: str = None,
            save_n_best: int = 5,
            resume: str = None):
        """
        :param logdir: log directory to use for checkpoint saving
        :param save_n_best: number of best checkpoiont to keep
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
                criterion=state._criterion,
                optimizer=state._optimizer,
                scheduler=state._scheduler)

            print("loaded checkpoint \"{}\" (epoch {})".format(
                filename, checkpoint["epoch"]))
        else:
            raise Exception("no checkpoint found at \"{}\"".format(filename))

    def save_checkpoint(
            self, logdir, checkpoint, is_best,
            save_n_best=5, main_metric="loss_main", minimize_metric=True):
        suffix = f"{checkpoint['stage']}.{checkpoint['epoch']}"
        filepath = UtilsFactory.save_checkpoint(
            logdir=logdir,
            checkpoint=checkpoint,
            is_best=is_best,
            suffix=suffix)
        checkpoint_metric = checkpoint["valid_metrics"].get(main_metric, None)
        checkpoint_metric = checkpoint_metric or checkpoint.get("epoch", -1)
        self.top_best_metrics.append((filepath, checkpoint_metric))
        self.top_best_metrics = sorted(
            self.top_best_metrics,
            key=lambda x: x[1],
            reverse=not minimize_metric)
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
            criterion=state._criterion,
            optimizer=state._optimizer,
            scheduler=state._scheduler,
            valid_metrics=dict(state.valid_metrics),  # @TODO: save defaultdict
            epoch_metrics=dict(state.epoch_metrics),  # @TODO: save defaultdict
            best_metrics=dict(state.best_metrics),  # @TODO: save defaultdict
            stage=state.stage,
            epoch=state.epoch)
        self.save_checkpoint(
            logdir=self.logdir,
            checkpoint=checkpoint,
            is_best=state.is_best_epoch,
            save_n_best=self.save_n_best,
            main_metric=state.main_metric,
            minimize_metric=state.minimize_metric)

    def on_train_end(self, state):
        print("Top best models:")
        top_best_metrics_str = "\n".join([
            "{filepath}\t{metric:.4f}".format(
                filepath=filepath, metric=metric)
            for filepath, metric in self.top_best_metrics
        ])
        print(top_best_metrics_str)


# @TODO: rewrite for optmizer_key
class OptimizerCallback(Callback):
    """
    Optimizer callback, abstraction over optimizer step.
    """

    def __init__(
            self,
            grad_clip: float = None,
            fp16_grad_scale: float = 128.0,
            accumulation_steps=1):
        """
        :param grad_clip: grap clipping specification kwargs
            @TODO: better support of different grad clip funcs
        :param fp16_grad_scale: grad scale for fp16 mode training
        """
        self.optimizer_wds = {}
        self.grad_clip = grad_clip
        self.fp16 = False
        self.fp16_grad_scale = fp16_grad_scale
        self.accumulation_steps = accumulation_steps
        self.accumulation_counter = 0

    def on_train_start(self, state):
        self.fp16 = isinstance(state.model, Fp16Wrap)

    def on_epoch_start(self, state):
        self.optimizer_wds = {}
        for key, optimizer_ in state._optimizer.items():
            wd = optimizer_.param_groups[0].get("weight_decay", 0.0)
            if wd > 0:
                self.optimizer_wds[key] = wd
                optimizer_.param_groups[0]["weight_decay"] = 0.0

    def grad_step(self, optimizer):
        for key, value in optimizer.items():
            if key in self.optimizer_wds:
                wd = self.optimizer_wds[key]
                for group in value.param_groups:
                    for param in group["params"]:
                        param.data = param.data.add(
                            -wd * group["lr"],
                            param.data)
                    if self.grad_clip is not None:
                        torch.nn.utils.clip_grad_norm_(
                            group["params"],
                            self.grad_clip)
            value.step()

    def on_batch_end(self, state):
        # @TODO: check this one
        if not state.is_train:
            return

        self.accumulation_counter += 1

        if not self.fp16:
            # for _, value in state._optimizer.items():
            #     value.zero_grad()

            # @TODO: check this one
            if len(state._optimizer) > 0:
                for key, value in state.loss.items():
                    value.backward()

                if (self.accumulation_counter + 1) \
                        % self.accumulation_steps == 0:
                    self.grad_step(state._optimizer)
                    state.model.zero_grad()
                    self.accumulation_counter = 0
        else:
            state.model.zero_grad()
            if len(state._optimizer) > 0:
                assert len(state._optimizer) == 1, \
                    "fp16 mode works only with one optimizer for now"

                for key, value in state.loss.items():
                    scaled_loss = self.fp16_grad_scale * value.float()
                    scaled_loss.backward()

                master_params = list(
                    state._optimizer["main"].param_groups[0]["params"])
                model_params = list(filter(
                    lambda p: p.requires_grad,
                    state.model.parameters()))

                copy_grads(source=model_params, target=master_params)

                for param in master_params:
                    param.grad.data.mul_(1. / self.fp16_grad_scale)

                self.grad_step(state._optimizer)

                copy_params(source=master_params, target=model_params)
                torch.cuda.synchronize()

    def on_epoch_end(self, state):
        for key, value in self.optimizer_wds.items():
            state._optimizer[key].param_groups[0]["weight_decay"] = value


class SchedulerCallback(Callback):
    def __init__(
            self,
            scheduler_key: str = "main",
            mode: str = "epoch",
            reduce_metric: str = None):
        self.scheduler_key = scheduler_key
        self.mode = mode
        self.reduce_metric = reduce_metric

    @staticmethod
    def get_momentum(optimizer):
        # @TODO: need better solutions
        if isinstance(optimizer, torch.optim.Adam):
            return list(optimizer.param_groups)[0]["betas"][0]
        elif isinstance(optimizer, torch.optim.SGD):
            return list(optimizer.param_groups)[0]["momentum"]
        else:
            return None

    def step(self, state):
        scheduler = state._scheduler[self.scheduler_key]
        if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
            scheduler.step(state.valid_metrics[self.reduce_metric])
            state.lr[self.scheduler_key] = \
                list(scheduler.optimizer.param_groups)[0]["lr"]
        else:
            scheduler.step()
            state.lr[self.scheduler_key] = scheduler.get_lr()[0]

        momentum = self.get_momentum(scheduler.optimizer)
        if isinstance(momentum, float):
            state.momentum[self.scheduler_key] = momentum

    def on_batch_end(self, state):
        if self.mode == "batch":
            self.step(state=state)

    def on_epoch_end(self, state):
        if self.mode == "epoch":
            self.step(state=state)


class LRUpdater(Callback):
    """Basic class that all Lr updaters inherit from"""

    def __init__(self, optimizer_key: str = "main"):
        """
        :param optimizer_key: which optimizer key to use
            for learning rate scheduling
        """
        self.init_lr = 0
        self.optimizer_key = optimizer_key

    def calc_lr(self):
        return None

    def calc_momentum(self):
        return None

    @staticmethod
    def update_lr(optimizer, new_lr):
        for pg in optimizer.param_groups:
            pg["lr"] = new_lr

    @staticmethod
    def update_momentum(optimizer, new_momentum):
        if "betas" in optimizer.param_groups[0]:
            for pg in optimizer.param_groups:
                pg["betas"] = (new_momentum, pg["betas"][1])
        else:
            for pg in optimizer.param_groups:
                pg["momentum"] = new_momentum

    def update_optimizer(self, state, optimizer):
        if state.is_train:
            new_lr = self.calc_lr()
            if new_lr is not None:
                self.update_lr(optimizer[self.optimizer_key], new_lr)
                state.lr[self.optimizer_key] = new_lr
            new_momentum = self.calc_momentum()
            if new_momentum is not None:
                self.update_momentum(
                    optimizer[self.optimizer_key],
                    new_momentum)
                state.momentum[self.optimizer_key] = new_momentum
        else:
            state.lr[self.optimizer_key] = 0
            state.momentum[self.optimizer_key] = 0

    def on_train_start(self, state):
        self.init_lr = state._optimizer[self.optimizer_key].defaults["lr"]

    def on_loader_start(self, state):
        self.update_optimizer(state=state, optimizer=state._optimizer)

    def on_batch_end(self, state):
        self.update_optimizer(state=state, optimizer=state._optimizer)


class OneCycleLR(LRUpdater):
    """
    An learning rate updater
        that implements the Circular Learning Rate (CLR) scheme.
    Learning rate is increased then decreased linearly.
    """

    def __init__(
            self,
            cycle_len: int,
            div: int,
            cut_div: int,
            momentum_range: Tuple[float, float],
            optimizer_key: str = "main"):
        """

        :param init_lr: init learning rate for torch optimizer
        :param cycle_len: (int) num epochs to apply one cycle policy
        :param div: (int) ratio between initial lr and maximum lr
        :param cut_div: (int) which part of cycle lr will grow
            (Ex: cut_div=4 -> 1/4 lr grow, 3/4 lr decrease
        :param momentum_range: (tuple(int, int)) max and min momentum values
        :param optimizer_key: which optimizer key to use
            for learning rate scheduling
        """
        super().__init__(optimizer_key=optimizer_key)
        self.total_iter = None
        self.div = div
        self.cut_div = cut_div
        self.cycle_iter = 0
        self.cycle_count = 0
        self.cycle_len = cycle_len
        # point in iterations for starting lr decreasing
        self.cut_point = None
        self.momentum_range = momentum_range

    def calc_lr(self):
        # calculate percent for learning rate change
        if self.cycle_iter > self.cut_point:
            percent = (1 - (self.cycle_iter - self.cut_point) /
                       (self.total_iter - self.cut_point))
        else:
            percent = self.cycle_iter / self.cut_point
        res = self.init_lr * (1 + percent * (self.div - 1)) / self.div

        self.cycle_iter += 1
        if self.cycle_iter == self.total_iter:
            self.cycle_iter = 0
            self.cycle_count += 1
        return res

    def calc_momentum(self):
        if self.cycle_iter > self.cut_point:
            now_ = (self.cycle_iter - self.cut_point)
            all_ = (self.total_iter - self.cut_point)
            percent = now_ / all_
        else:
            percent = 1 - self.cycle_iter / self.cut_point
        res = (self.momentum_range[1] +
               percent * (self.momentum_range[0] - self.momentum_range[1]))
        return res

    def on_loader_start(self, state):
        if state.is_train:
            self.total_iter = state.loader_len * self.cycle_len
            self.cut_point = self.total_iter // self.cut_div

        super().on_loader_start(state=state)


class LRFinder(LRUpdater):
    """
    Helps you find an optimal learning rate for a model,
        as per suggetion of 2015 CLR paper.
    Learning rate is increased in linear or log scale, depending on user input.

    https://sgugger.github.io/how-do-you-find-a-good-learning-rate.html
    """

    def __init__(self, final_lr, n_steps=None, optimizer_key="main"):
        """

        :param init_lr: initial learning rate to use
        :param final_lr: final learning rate to try with
        :param n_steps:  number of batches to try;
            if None - whole loader would be used.
        :param optimizer_key: which optimizer key to use
            for learning rate scheduling
        """
        super().__init__(optimizer_key=optimizer_key)

        self.final_lr = final_lr
        self.n_steps = n_steps
        self.multiplier = 0
        self.find_iter = 0

    def calc_lr(self):
        res = self.init_lr * self.multiplier ** self.find_iter
        self.find_iter += 1
        return res

    def on_batch_end(self, state):
        super().on_batch_end(state=state)
        if self.find_iter > self.n_steps:
            raise NotImplementedError("End of LRFinder")

    def on_loader_start(self, state):
        if state.is_train:
            lr_ = self.final_lr / self.init_lr
            self.n_steps = self.n_steps or len(state.loader)
            self.multiplier = lr_ ** (1 / self.n_steps)

        super().on_loader_start(state=state)


class ClassificationLossCallback(Callback):
    def __init__(
            self,
            input_key: str = "targets",
            output_key: str = "logits"):
        self.input_key = input_key
        self.output_key = output_key

    def on_batch_end(self, state):
        state.loss["main"] = state._criterion["main"](
            state.output[self.output_key],
            state.input[self.input_key])


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
                        suffix=".".join([state.loader_mode, key])), value)


class MixupCallback(Callback):
    def __init__(
            self,
            mixup_keys: List[str],
            alpha: float,
            share_lambda: bool = True):
        self.mixup_keys = mixup_keys
        self.alpha = alpha
        self.share_lambda = share_lambda

    def on_batch_start(self, state):
        lambda_ = compute_mixup_lambda(
            state.batch_size, self.alpha, self.share_lambda)
        for key in self.mixup_keys:
            state.input[key] = mixup_torch(state.input[key], lambda_=lambda_)
