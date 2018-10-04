import logging
import os
import time
import numpy as np
from collections import defaultdict
from typing import Tuple, List, Dict
import torch
from torchnet import meter
from tensorboardX import SummaryWriter

from prometheus.utils.metrics import precision
from prometheus.utils.fp16 import Fp16Wrap, copy_params, copy_grads
from prometheus.utils.factory import UtilsFactory
from prometheus.utils.misc import FakeSummaryWriter


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

    def on_stage_init(self, model, stage):
        pass

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

    def on_stage_init(self, model, stage):
        for key, value in self.callbacks.items():
            value.on_stage_init(model=model, stage=stage)

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


class LoggerCallback(Callback):
    """
    Logger callback, translates state.*_metrics to console and tensorboard
    """

    def __init__(self,
                 tensorboard_loggers: Dict[str, SummaryWriter] = None,
                 reset_step: bool = False,
                 verbose=False):
        """

        :param tensorboard_loggers: TF Summary loggers used
            during train/infer/debug.
            If none, only console logging will be used.
        :param reset_step: boolean flag;
            if False - logs will be combined for train/valid
            if True  - logs will be separated
        """
        # @TODO: make loggers autocreation based on loader name
        self.loggers = (tensorboard_loggers
                        or defaultdict(lambda: FakeSummaryWriter()))
        self.reset_step = reset_step
        self.verbose = verbose

    def on_epoch_start(self, state):
        state.epoch_metrics = defaultdict(
            lambda: defaultdict(lambda: meter.AverageValueMeter()))

    def on_loader_start(self, state):
        lm = state.loader_mode
        self.time = time.time()
        state.step = (state.step
                      or state.epoch * len(state.loader) * state.batch_size)
        state.epoch_metrics[lm] = defaultdict(
            lambda: meter.AverageValueMeter())

    def on_batch_start(self, state):
        state.batch_metrics = defaultdict(lambda: 0)
        state.batch_metrics["data time"] = time.time() - self.time

    def on_batch_end(self, state):
        lm = state.loader_mode
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

        for key, value in state.batch_metrics.items():
            # @TODO: batch metrics print?
            if isinstance(value, (tuple, list)):
                for i, value_ in enumerate(value):
                    key_ = f"key_{i}"
                    state.epoch_metrics[lm][key_].add(value_)
                    self.loggers[lm].add_scalar(key_, value_, state.step)
            else:
                state.epoch_metrics[lm][key].add(value)
                self.loggers[lm].add_scalar(key, value, state.step)

        self.time = time.time()
        state.step += bs

    def on_loader_end(self, state):
        lm = state.loader_mode

        state.epoch_metrics[lm] = {
            key: UtilsFactory.get_val_from_metric(value)
            for key, value in state.epoch_metrics[lm].items()
        }

        for key, value in state.epoch_metrics[lm].items():
            self.loggers[lm].add_scalar(f"epoch {key}", value, state.epoch)

        if self.verbose:
            epoch_metrics_str = "\t".join([
                "{key} {value:.4f}".format(key=key, value=value)
                for key, value in sorted(state.epoch_metrics[lm].items())
            ])

            print("{epoch} * Epoch ({mode}): {metrics}".format(
                epoch=state.epoch, mode=lm, metrics=epoch_metrics_str))

        if self.reset_step:
            state.step = None


class Logger(Callback):
    def __init__(
            self,
            logdir: str = None,
    ):
        """
        :param logdir: log directory to use for checkpoint saving
        """
        super().__init__()
        os.makedirs(logdir, exist_ok=True)
        log_filepath = os.path.join(logdir, 'txt_logs.txt')
        self.logger = self._get_logger(log_filepath)

    def on_epoch_begin(self, state):
        self.logger.info('Epoch {} | optimizer "{}" | lr {}'.format(
            state.epoch, state.optimizer.__class__.__name__, state.lr))

    def on_epoch_end(self, state):
        for postifx, (k, v) in zip(["", "\n"], state.epoch_metrics.items()):
            self.logger.info(f"{k} metrics: " + self._get_metrics_string(v) +
                             postifx)

    def on_train_begin(self, state):
        self.logger.info(
            'Starting training with params:\n{}\n\n'.format(state))

    def on_batch_end(self, state):
        if state.is_train:
            # print(self.state.epoch_metrics['train'])
            state.loader.set_postfix(**{
                k: "{:.5f}".format(v)
                for k, v in state.batch_metrics.items()
            })

    @staticmethod
    def _get_logger(log_filepath):
        logger = logging.getLogger(log_filepath)
        logger.setLevel(logging.DEBUG)
        fh = logging.FileHandler(log_filepath)
        fh.setLevel(logging.INFO)
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)
        formatter = logging.Formatter('[%(asctime)s] %(message)s')
        fh.setFormatter(formatter)
        ch.setFormatter(formatter)
        # add the handlers to the logger
        logger.addHandler(fh)
        logger.addHandler(ch)
        return logger

    def _get_metrics_string(self, metrics):
        return " | ".join(
            "{}: {:.5f}".format(k, v) for k, v in metrics.items())


class PrecisionCallback(Callback):
    """
    Precision metric callback.
    """

    def __init__(self,
                 input_key: str = "targets",
                 output_key: str = "logits",
                 precision_args: List[int] = None):
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
        super().__init__()
        self.input_key = input_key
        self.output_key = output_key
        self.precision_args = precision_args or [1, 3, 5]

    def on_batch_end(self, state):
        prec = precision(
            state.output[self.output_key],
            state.input[self.input_key],
            topk=self.precision_args)

        for p, metric in zip(self.precision_args, prec):
            key = "precision{:02}".format(p)
            metric_ = metric.item()
            state.batch_metrics[key] = metric_


class CheckpointCallback(Callback):
    """
    Checkpoint callback to save/restore your mode/criterion/optimizer/metrics.
    """

    def __init__(self,
                 logdir: str = None,
                 save_n_best: int = 5,
                 resume: str = None,
                 main_metric: str = "loss_main",
                 minimize: bool = True,
                 valid_loader: str = "valid"):
        """
        :param logdir: log directory to use for checkpoint saving
        :param save_n_best: number of best checkpoiont to keep
        :param resume: path to checkpoint to load and initialize runner state
        :param main_metric: which metric to use for checkpoint comparison
        :param minimize: boolean flag if we need to minimize or maximize metric
        """
        self.logdir = logdir
        self.save_n_best = save_n_best
        self.resume = resume
        self.main_metric = main_metric
        self.minimize = minimize
        self.valid_loader = valid_loader
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

    def save_checkpoint(self, logdir, checkpoint, is_best, save_n_best=5):
        suffix = f"{checkpoint['stage']}.{checkpoint['epoch']}"
        filepath = UtilsFactory.save_checkpoint(
            logdir=logdir,
            checkpoint=checkpoint,
            is_best=is_best,
            suffix=suffix)
        self.top_best_metrics.append(
            (filepath, checkpoint["valid_metrics"][self.main_metric]))
        self.top_best_metrics = sorted(
            self.top_best_metrics,
            key=lambda x: x[1],
            reverse=not self.minimize)
        if len(self.top_best_metrics) > save_n_best:
            last_item = self.top_best_metrics.pop(-1)
            last_filepath = last_item[0]
            os.remove(last_filepath)

    def pack_checkpoint(self, **kwargs):
        return UtilsFactory.pack_checkpoint(**kwargs)

    @staticmethod
    def process_epoch_metrics(epoch_metrics,
                              best_metrics,
                              valid_loader="valid",
                              main_metric="loss",
                              minimize=True):
        valid_metrics = epoch_metrics[valid_loader]
        is_best = True \
            if best_metrics is None \
            else (minimize != (
                valid_metrics[main_metric] > best_metrics[main_metric]))
        best_metrics = valid_metrics if is_best else best_metrics
        return best_metrics, valid_metrics, is_best

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
        if state.loader_mode != self.valid_loader:
            return

        best_metrics, valid_metrics, is_best = self.process_epoch_metrics(
            state.epoch_metrics,
            state.best_metrics,
            valid_loader=self.valid_loader,
            main_metric=self.main_metric,
            minimize=self.minimize)
        valid_metrics = {
            key: value
            for key, value in valid_metrics.items()
            if isinstance(value, float)
        }
        state.best_metrics = {
            key: value
            for key, value in best_metrics.items() if isinstance(value, float)
        }
        state.valid_metrics = valid_metrics

        checkpoint = self.pack_checkpoint(
            model=state.model,
            criterion=state._criterion,
            optimizer=state._optimizer,
            scheduler=state._scheduler,
            valid_metrics=dict(valid_metrics),  # @TODO: save defaultdict
            epoch_metrics=dict(state.epoch_metrics),  # @TODO: save defaultdict
            best_metrics=dict(state.best_metrics),  # @TODO: save defaultdict
            stage=state.stage,
            epoch=state.epoch)
        self.save_checkpoint(
            logdir=self.logdir,
            checkpoint=checkpoint,
            is_best=is_best,
            save_n_best=self.save_n_best)

    def on_train_end(self, state):
        print("Top best models:")
        top_best_metrics_str = "\n".join([
            "{filepath}\t{metric:.4f}".format(
                filepath=filepath, metric=metric)
            for filepath, metric in self.top_best_metrics
        ])
        print(top_best_metrics_str)


class OptimizerCallback(Callback):
    """
    Optimizer callback, abstraction over optimizer step.
    """

    def __init__(self, grad_clip: float = None,
                 fp16_grad_scale: float = 128.0):
        """
        :param grad_clip: grap clipping specification kwargs
            @TODO: better support of different grad clip funcs
        :param fp16_grad_scale: grad scale for fp16 mode training
        """
        self.optimizer_wds = {}
        self.grad_clip = grad_clip
        self.fp16 = False
        self.fp16_grad_scale = fp16_grad_scale

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
                        param.data = param.data.add(-wd * group["lr"],
                                                    param.data)
                    if self.grad_clip is not None:
                        torch.nn.utils.clip_grad_norm_(group["params"],
                                                       self.grad_clip)
            value.step()

    def on_batch_end(self, state):
        if not state.is_train:
            return

        if not self.fp16:
            for _, value in state._optimizer.items():
                value.zero_grad()

            if len(state._optimizer) > 0:
                for key, value in state.loss.items():
                    value.backward()
                self.grad_step(state._optimizer)
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
                model_params = list(
                    filter(lambda p: p.requires_grad,
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
    def __init__(self,
                 scheduler_key: str = "main",
                 mode: str = "epoch",
                 reduce_metric: str = None):
        self.scheduler_key = scheduler_key
        self.mode = mode
        self.reduce_metric = reduce_metric

    def step(self, state):
        scheduler = state._scheduler[self.scheduler_key]
        if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
            scheduler.step(state.valid_metrics[self.reduce_metric])
            state.lr[self.scheduler_key] = \
                list(scheduler.optimizer.param_groups)[0]["lr"]
        else:
            scheduler.step()
            state.lr[self.scheduler_key] = scheduler.get_lr()[0]
        state.momentum[self.scheduler_key] = \
            list(scheduler.optimizer.param_groups)[0]["betas"][0]

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
                self.update_momentum(optimizer[self.optimizer_key],
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

    def __init__(self,
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
            self.total_iter = len(state.loader) * self.cycle_len
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
        res = self.init_lr * self.multiplier**self.find_iter
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
            self.multiplier = lr_**(1 / self.n_steps)

        super().on_loader_start(state=state)


class ClassificationLossCallback(Callback):
    def on_batch_end(self, state):
        state.loss["main"] = state._criterion["main"](state.output["logits"],
                                                      state.input["targets"])


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
