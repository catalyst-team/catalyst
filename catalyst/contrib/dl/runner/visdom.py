from collections import Counter
import logging
import threading
from typing import Dict, Union
import queue

from alchemy.logger import validate, validate_metric, Logger

from catalyst.dl import utils
from catalyst.dl.core import Experiment, Runner
from catalyst.dl.runner import SupervisedRunner
from catalyst.contrib.dl.runner.alchemy import AlchemyRunner

import visdom


logger = logging.getLogger(__name__)


class VisdomLogger(Logger):
    def __init__(
        self,
        experiment: str,
        batch_size: int = None,
        server: str = "localhost",
        port: int = 8097,
    ):
        self._batch_size = max(int(batch_size or int(1e3)), 1)
        self._experiment = experiment
        self._server = server
        self._port = port
        self._counters = Counter()
        self._queue = queue.Queue()
        self._thread = threading.Thread(target=self._run_worker)
        self._thread.start()
        try:
            self.viz = visdom.Visdom(
                server=self._server, port=self._port, env=self._experiment
            )
            startup_sec = 1
            while not self.viz.check_connection() and startup_sec > 0:
                time.sleep(0.1)
                startup_sec -= 0.1
            assert self.viz.check_connection(), "No connection could be formed quickly"
        except BaseException as e:
            logger.error(
                "The visdom experienced an exception while"
                + "running: {}".format(repr(e))
            )

    def _run_worker(self):
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

    def plot_lines(self, batch):
        for msg in batch:
            opts = dict(
                xlabel="epochs",
                legend=['train', 'valid'],
                ylabel=msg['name'],
                title=msg['name'])
            self.viz.line(
                X=[self._counters[msg['full_name']]],
                Y=[msg['value']],
                win=msg['name'],
                name=msg['mode'],
                update='append', opts=opts)

    def log_scalar(
        self,
        name: str,
        mode: str,
        full_name: str,
        value: Union[int, float],
    ):
        self._queue.put(
            dict(
                name=validate_metric(name, f"invalid metric name: {name}"),
                full_name=full_name,
                mode=mode,
                value=value,
                step=self._counters[full_name],
            )
        )
        self._counters[full_name] += 1


class VisdomRunner(AlchemyRunner):

    def _log_metrics(self, metrics: Dict, mode: str, suffix: str = ""):
        for key, value in metrics.items():
            metric_name = f"{key}{suffix}"
            full_metric_name = f"{mode}/{metric_name}"
            self.logger.log_scalar(metric_name, mode, full_metric_name, value)

    def _pre_experiment_hook(self, experiment: Experiment):
        monitoring_params = experiment.monitoring_params

        log_on_batch_end: bool = \
            monitoring_params.pop("log_on_batch_end", False)
        log_on_epoch_end: bool = \
            monitoring_params.pop("log_on_epoch_end", True)

        self._init(
            log_on_batch_end=log_on_batch_end,
            log_on_epoch_end=log_on_epoch_end,
        )
        self.logger = VisdomLogger(**monitoring_params)


class SupervisedVisdomRunner(VisdomRunner, SupervisedRunner):
    """SupervisedRunner with Alchemy"""
    pass


__all__ = ["VisdomRunner", "SupervisedVisdomRunner"]
