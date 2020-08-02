# flake8: noqa
# @TODO: code formatting issue for 20.07 release
from typing import Dict, List, Union
from collections import Counter
import logging
import queue
import threading
import time

from alchemy.logger import Logger
import visdom

from catalyst.core.callback import (
    Callback,
    CallbackNode,
    CallbackOrder,
    CallbackScope,
)
from catalyst.core.runner import IRunner


class Visdom(Logger):
    """Logger, translates ``runner.*_metrics`` to Visdom.
    Read about Visdom here https://github.com/facebookresearch/visdom

    Example:
        .. code-block:: python

            VisdomLogger(
                env_name="...", # enviroment name
                server="localhost", # visdom server name
                port=8097, # visdom server port
            )
    """

    def __init__(
        self,
        env_name: str,
        batch_size: int = None,
        server: str = "localhost",
        port: int = 8097,
        log_to_filename: str = None,
        username: str = None,
        password: str = None,
    ):
        """
        Args:
            env_name (str): Environment name to plot to when
                no env is provided (default: main)
            batch_size (int): batch_size for log_on_batch_end
            server (str): the hostname of your
                visdom server (default: 'http://localhost')
            port (str): the port for your visdom server (default: 8097)
            log_to_filename (str): logs per-epoch metrics if set True
            username (str): username to use for authentication,
                if server started with -enable_login (default: None)
            password (str): password to use for authentication,
                if server started with -enable_login (default: None)
        """
        self._batch_size = max(int(batch_size or int(1e3)), 1)
        self._counters = Counter()
        self._queue = queue.Queue()
        self._thread = threading.Thread(target=self._run_worker)
        self._thread.start()
        try:
            self.viz = visdom.Visdom(
                server=server,
                port=port,
                env=env_name,
                log_to_filename=log_to_filename,
                username=username,
                password=password,
            )
            startup_sec = 1
            while not self.viz.check_connection() and startup_sec > 0:
                time.sleep(0.1)
                startup_sec -= 0.1
            assert (
                self.viz.check_connection()
            ), "No connection could be formed quickly"
        except Exception as e:
            logging.error(
                "The visdom experienced an exception while"
                + "running: {}".format(repr(e))  # noqa: P101
            )

    def _run_worker(self):
        """Runs worker to gather batch statistics."""
        running = True
        while running:
            batch = []
            try:
                while len(batch) < self._batch_size:
                    if batch:
                        msg = self._queue.get_nowait()
                    else:
                        msg = self._queue.get()
                    if msg is None:
                        running = False
                        break
                    batch.append(msg)
            except queue.Empty:
                pass
            if batch:
                self.plot_lines(batch)

    def plot_lines(self, batch: List[Dict]):
        """Plots vales from batch statistics.

        Args:
            batch (List[Dict]): List with dictionaries from log_scalar
        """
        for msg in batch:
            opts = {
                "xlabel": "epochs",
                "legend": ["train", "valid"],
                "ylabel": msg["name"],
                "title": msg["name"],
            }
            self.viz.line(
                X=[self._counters[msg["full_name"]]],
                Y=[msg["value"]],
                win=msg["name"],
                name=msg["mode"],
                update="append",
                opts=opts,
            )

    def log_scalar(
        self, name: str, mode: str, full_name: str, value: Union[int, float],
    ):
        """Logs scalar.

        Args:
            name (str): Environment name to plot to when
                no env is provided (default: main)
            mode (str): Metric's mode (example: train)
            full_name (str): Full metric name
            value (Union[int, float]): Metric's value
        """
        self._queue.put(
            {
                "name": name,
                "full_name": full_name,
                "mode": mode,
                "value": value,
                "step": self._counters[full_name],
            }
        )
        self._counters[full_name] += 1


class VisdomLogger(Callback):
    """Logger callback, translates ``runner.*_metrics`` to Visdom.
    Read about Visdom here https://github.com/facebookresearch/visdom

    Example:
        .. code-block:: python

            from catalyst.dl import SupervisedRunner, VisdomLogger

            runner = SupervisedRunner()

            runner.train(
                model=model,
                criterion=criterion,
                optimizer=optimizer,
                loaders=loaders,
                logdir=logdir,
                num_epochs=num_epochs,
                verbose=True,
                callbacks={
                    "logger": VisdomLogger(
                        env_name="...", # enviroment name
                        server="localhost", # visdom server name
                        port=8097, # visdom server port
                    )
                }
            )
    """

    def __init__(
        self,
        metric_names: List[str] = None,
        log_on_batch_end: bool = False,
        log_on_epoch_end: bool = True,
        **logging_params,
    ):
        """
        Args:
            metric_names (List[str]): list of metric names to log,
                if none - logs everything
            log_on_batch_end (bool): logs per-batch metrics if set True
            log_on_epoch_end (bool): logs per-epoch metrics if set True
        """
        super().__init__(
            order=CallbackOrder.logging,
            node=CallbackNode.master,
            scope=CallbackScope.experiment,
        )
        self.metrics_to_log = metric_names
        self.log_on_batch_end = log_on_batch_end
        self.log_on_epoch_end = log_on_epoch_end

        if not (self.log_on_batch_end or self.log_on_epoch_end):
            raise ValueError("You have to log something!")

        if (self.log_on_batch_end and not self.log_on_epoch_end) or (
            not self.log_on_batch_end and self.log_on_epoch_end
        ):
            self.batch_log_suffix = ""
            self.epoch_log_suffix = ""
        else:
            self.batch_log_suffix = "_batch"
            self.epoch_log_suffix = "_epoch"
        self.logger = Visdom(**logging_params)

    def _log_metrics(
        self, metrics: Dict[str, float], step: int, mode: str, suffix=""
    ):
        """Translate batch metrics to Visdom logger.

        Args:
            metrics (Dict[str, float]): Metrics from Catalyst
            step (int): Iteration step from Catalyst
            mode (str): Metric's mode (example: train)
            suffix (str): Additional suffix
        """
        if self.metrics_to_log is None:
            metrics_to_log = sorted(metrics.keys())
        else:
            metrics_to_log = self.metrics_to_log

        for name in metrics_to_log:
            if name in metrics:
                # Renaming catalyst metric names to visdom formatting
                real_mode = name.split("_")[0]
                splitted_name = name.split(real_mode + "_")[-1]
                metric_name = f"{splitted_name}{suffix}"
                full_metric_name = f"{real_mode}/{metric_name}"
                metric_value = metrics[name]
                # Log values
                self.logger.log_scalar(
                    metric_name, real_mode, full_metric_name, metric_value
                )

    def __del__(self):
        """@TODO: Docs. Contribution is welcome."""
        self.logger.close()

    def on_batch_end(self, runner: IRunner):
        """Translate batch metrics to Visdom."""
        if self.log_on_batch_end:
            mode = runner.loader_name
            metrics = runner.batch_metrics
            self._log_metrics(
                metrics=metrics,
                step=runner.global_sample_step,
                mode=mode,
                suffix=self.batch_log_suffix,
            )

    def on_epoch_end(self, runner: IRunner):
        """Translate epoch metrics to Visdom."""
        if self.log_on_epoch_end:
            self._log_metrics(
                metrics=runner.epoch_metrics,
                step=runner.global_epoch,
                mode=runner.loader_name,
                suffix=self.epoch_log_suffix,
            )


__all__ = ["VisdomLogger"]
