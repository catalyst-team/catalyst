import os
from typing import Dict
from collections import OrderedDict

from pathlib import Path
import safitty

from catalyst.dl import utils
from catalyst.dl.core import Callback, CallbackOrder, RunnerState
from catalyst.utils import is_exception


class BaseCheckpointCallback(Callback):
    """
    Base class for all checkpoint callbacks
    """
    def __init__(self, metric_filename: str = "_metrics.json"):
        """
        Args:
            metric_filename (str): filename to save metrics
                in checkpoint folder. Must ends on ``.json`` or ``.yml``
        """
        super().__init__(CallbackOrder.Logger)
        self.metric_filename = metric_filename
        self.metrics: dict = {}

    def get_metric(self, **kwargs) -> Dict:
        return self.metrics

    def save_metric(self, logdir: str, metrics: Dict) -> None:
        safitty.save(metrics, f"{logdir}/checkpoints/{self.metric_filename}")

    def truncate_checkpoints(self, **kwargs) -> None:
        pass

    def process_checkpoint(self, **kwargs) -> None:
        pass

    def get_checkpoint_suffix(self, checkpoint: dict) -> str:
        pass

    def on_exception(self, state: RunnerState):
        exception = state.exception
        if not is_exception(exception):
            return

        try:
            valid_metrics = state.metrics.valid_values
            epoch_metrics = state.metrics.epoch_values
            checkpoint = utils.pack_checkpoint(
                model=state.model,
                criterion=state.criterion,
                optimizer=state.optimizer,
                scheduler=state.scheduler,
                epoch_metrics=epoch_metrics,
                valid_metrics=valid_metrics,
                stage=state.stage,
                epoch=state.epoch,
                checkpoint_data=state.checkpoint_data
            )
            suffix = self.get_checkpoint_suffix(checkpoint)
            suffix = f"{suffix}.exception_{exception.__class__.__name__}"
            utils.save_checkpoint(
                logdir=f"{state.logdir}/checkpoints/",
                checkpoint=checkpoint,
                suffix=suffix,
                is_best=False,
                is_last=False
            )
            metrics = self.metrics
            metrics[suffix] = valid_metrics
            self.save_metric(state.logdir, metrics)
        except Exception:
            pass


class CheckpointCallback(BaseCheckpointCallback):
    """
    Checkpoint callback to save/restore your model/criterion/optimizer/metrics.
    """
    def __init__(
        self,
        save_n_best: int = 1,
        resume: str = None,
        resume_dir: str = None,
        metric_filename: str = "_metrics.json"
    ):
        """
        Args:
            save_n_best (int): number of best checkpoint to keep
            resume (str): path to checkpoint to load
                and initialize runner state
            metric_filename (str): filename to save metrics
                in checkpoint folder. Must ends on ``.json`` or ``.yml``
        """
        super().__init__(metric_filename)
        self.save_n_best = save_n_best
        self.resume = resume
        self.resume_dir = resume_dir
        self.top_best_metrics = []

        self._keys_from_state = ["resume", "resume_dir"]

    def get_checkpoint_suffix(self, checkpoint: dict) -> str:
        result = f"{checkpoint['stage']}.{checkpoint['epoch']}"
        return result

    @staticmethod
    def load_checkpoint(*, filename, state: RunnerState):
        if os.path.isfile(filename):
            print(f"=> loading checkpoint {filename}")
            checkpoint = utils.load_checkpoint(filename)

            state.epoch = checkpoint["epoch"]

            utils.unpack_checkpoint(
                checkpoint,
                model=state.model,
                criterion=state.criterion,
                optimizer=state.optimizer,
                scheduler=state.scheduler
            )

            print(
                f"loaded checkpoint {filename} (epoch {checkpoint['epoch']})"
            )
        else:
            raise Exception(f"No checkpoint found at {filename}")

    def get_metric(self, last_valid_metrics) -> Dict:
        checkpoints = [
            (Path(filepath).stem, valid_metric)
            for (filepath, _, valid_metric) in self.top_best_metrics
        ]
        best_valid_metrics = checkpoints[0][1]
        metrics = OrderedDict(
            [("best", best_valid_metrics)] +
            checkpoints +
            [("last", last_valid_metrics)]
        )

        self.metrics = metrics
        return self.metrics

    def truncate_checkpoints(self, minimize_metric: bool) -> None:
        self.top_best_metrics = sorted(
            self.top_best_metrics,
            key=lambda x: x[1],
            reverse=not minimize_metric
        )
        if len(self.top_best_metrics) > self.save_n_best:
            last_item = self.top_best_metrics.pop(-1)
            last_filepath = Path(last_item[0])
            last_filepaths = last_filepath.parent.glob(
                last_filepath.name.replace(".pth", "*"))
            for filepath in last_filepaths:
                os.remove(filepath)

    def process_checkpoint(
        self,
        logdir: str,
        checkpoint: Dict,
        is_best: bool,
        main_metric: str = "loss",
        minimize_metric: bool = True
    ):
        suffix = self.get_checkpoint_suffix(checkpoint)
        utils.save_checkpoint(
            logdir=f"{logdir}/checkpoints/",
            checkpoint=checkpoint,
            suffix=f"{suffix}_full",
            is_best=is_best,
            is_last=True,
            special_suffix="_full"
        )

        exclude = ["criterion", "optimizer", "scheduler"]
        checkpoint = {
            key: value
            for key, value in checkpoint.items()
            if all(z not in key for z in exclude)
        }
        filepath = utils.save_checkpoint(
            checkpoint=checkpoint,
            logdir=f"{logdir}/checkpoints/",
            suffix=suffix,
            is_best=is_best,
            is_last=True
        )

        valid_metrics = checkpoint["valid_metrics"]
        checkpoint_metric = valid_metrics[main_metric]
        self.top_best_metrics.append(
            (filepath, checkpoint_metric, valid_metrics)
        )
        self.truncate_checkpoints(minimize_metric=minimize_metric)

        metrics = self.get_metric(valid_metrics)
        self.save_metric(logdir, metrics)

    def on_stage_start(self, state: RunnerState):
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

        valid_metrics = dict(state.metrics.valid_values)
        epoch_metrics = dict(state.metrics.epoch_values)

        checkpoint = utils.pack_checkpoint(
            model=state.model,
            criterion=state.criterion,
            optimizer=state.optimizer,
            scheduler=state.scheduler,
            epoch_metrics=epoch_metrics,
            valid_metrics=valid_metrics,
            stage=state.stage,
            epoch=state.epoch,
            checkpoint_data=state.checkpoint_data
        )
        self.process_checkpoint(
            logdir=state.logdir,
            checkpoint=checkpoint,
            is_best=state.metrics.is_best,
            main_metric=state.main_metric,
            minimize_metric=state.minimize_metric
        )

    def on_stage_end(self, state: RunnerState):
        print("Top best models:")
        top_best_metrics_str = "\n".join(
            [
                "{filepath}\t{metric:3.4f}".format(
                    filepath=filepath, metric=checkpoint_metric
                ) for filepath, checkpoint_metric, _ in self.top_best_metrics
            ]
        )
        print(top_best_metrics_str)


class IterationCheckpointCallback(BaseCheckpointCallback):
    """
    Iteration checkpoint callback to save your model/criterion/optimizer
    """
    def __init__(
        self,
        save_n_last: int = 3,
        num_iters: int = 100,
        stage_restart: bool = True,
        metric_filename: str = "_metrics_iter.json"
    ):
        """
        Args:
            save_n_last (int): number of last checkpoint to keep
            num_iters (int): save the checkpoint every `num_iters`
            stage_restart (bool): restart counter every stage or not
            metric_filename (str): filename to save metrics
                in checkpoint folder. Must ends on ``.json`` or ``.yml``
        """
        super().__init__(metric_filename)
        self.save_n_last = save_n_last
        self.num_iters = num_iters
        self.stage_restart = stage_restart
        self._iteration_counter = 0
        self.last_checkpoints = []

    def get_checkpoint_suffix(self, checkpoint: dict) -> str:
        result = f"{checkpoint['stage']}." \
                 f"epoch.{checkpoint['epoch']}." \
                 f"iter.{self._iteration_counter}"

        return result

    def get_metric(self, **kwargs) -> Dict:
        checkpoints = [
            (Path(filepath).stem, batch_values)
            for (filepath, batch_values) in self.last_checkpoints
        ]

        self.metrics = OrderedDict(checkpoints)
        return self.metrics

    def truncate_checkpoints(self, **kwargs) -> None:
        if len(self.last_checkpoints) > self.save_n_last:
            item = self.last_checkpoints.pop(0)
            top_filepath = item[0]
            os.remove(top_filepath)

    def process_checkpoint(
        self,
        logdir: str,
        checkpoint: Dict,
        batch_values: Dict[str, float]
    ):
        filepath = utils.save_checkpoint(
            logdir=f"{logdir}/checkpoints/",
            checkpoint=checkpoint,
            suffix=self.get_checkpoint_suffix(checkpoint),
            is_best=False,
            is_last=False
        )

        self.last_checkpoints.append((filepath, batch_values))
        self.truncate_checkpoints()

        metrics = self.get_metric()
        self.save_metric(logdir, metrics)
        print(f"\nSaved checkpoint at {filepath}")

    def on_stage_start(self, state):
        if self.stage_restart:
            self._iteration_counter = 0

    def on_batch_end(self, state):
        self._iteration_counter += 1
        if self._iteration_counter % self.num_iters == 0:
            checkpoint = utils.pack_checkpoint(
                model=state.model,
                criterion=state.criterion,
                optimizer=state.optimizer,
                scheduler=state.scheduler,
                epoch_metrics=None,
                valid_metrics=None,
                stage=state.stage,
                epoch=state.epoch
            )
            self.process_checkpoint(
                logdir=state.logdir,
                checkpoint=checkpoint,
                batch_values=state.metrics.batch_values
            )


__all__ = ["CheckpointCallback", "IterationCheckpointCallback"]
