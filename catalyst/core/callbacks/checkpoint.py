from typing import Dict, Tuple, Union
from collections import OrderedDict
import os
from pathlib import Path

from catalyst.core import Callback, CallbackNode, CallbackOrder, State, utils


def _pack_state(state: State):
    checkpoint = utils.pack_checkpoint(
        model=state.model,
        criterion=state.criterion,
        optimizer=state.optimizer,
        scheduler=state.scheduler,
        epoch_metrics=dict(state.epoch_metrics),
        valid_metrics=dict(state.valid_metrics),
        stage_name=state.stage_name,
        epoch=state.epoch,
        loader_name=state.loader_name,
        loader_step=state.loader_step,
        global_epoch=state.global_epoch,
        checkpoint_data=state.checkpoint_data,
        main_metric=state.main_metric,
        minimize_metric=state.minimize_metric,
        valid_loader=state.valid_loader,
    )
    return checkpoint


def _load_checkpoint(*, filename, state: State, load_full: bool = True):
    if not os.path.isfile(filename):
        raise Exception(f"No checkpoint found at {filename}")

    print(f"=> loading checkpoint {filename}")
    checkpoint = utils.load_checkpoint(filename)

    if not state.stage_name.startswith("infer") and load_full:
        state.stage_name = checkpoint["stage_name"]
        state.epoch = checkpoint["epoch"]
        state.global_epoch = checkpoint["global_epoch"]
        # @TODO: should we also load,
        # checkpoint_data, main_metric, minimize_metric, valid_loader ?
        # epoch_metrics, valid_metrics ?

    if load_full:
        utils.unpack_checkpoint(
            checkpoint,
            model=state.model,
            criterion=state.criterion,
            optimizer=state.optimizer,
            scheduler=state.scheduler,
        )

        print(
            f"loaded state checkpoint {filename} "
            f"(global epoch {checkpoint['global_epoch']}, "
            f"epoch {checkpoint['epoch']}, "
            f"stage {checkpoint['stage_name']})"
        )
    else:
        utils.unpack_checkpoint(
            checkpoint, model=state.model,
        )

        print(f"loaded model checkpoint {filename}")


class BaseCheckpointCallback(Callback):
    """Base class for all checkpoint callbacks."""

    def __init__(self, metrics_filename: str = "_metrics.json"):
        """
        Args:
            metrics_filename (str): filename to save metrics
                in checkpoint folder. Must ends on ``.json`` or ``.yml``
        """
        super().__init__(order=CallbackOrder.External, node=CallbackNode.All)
        self.metrics_filename = metrics_filename
        self.metrics: dict = {}

    def get_checkpoint_suffix(self, checkpoint: dict) -> str:
        return "checkpoint"

    def save_metric(self, logdir: Union[str, Path], metrics: Dict) -> None:
        utils.save_config(
            metrics, f"{logdir}/checkpoints/{self.metrics_filename}"
        )

    def on_exception(self, state: State):
        exception = state.exception
        if not utils.is_exception(exception):
            return

        try:
            checkpoint = _pack_state(state)
            suffix = self.get_checkpoint_suffix(checkpoint)
            suffix = f"{suffix}.exception_{exception.__class__.__name__}"
            utils.save_checkpoint(
                logdir=Path(f"{state.logdir}/checkpoints/"),
                checkpoint=checkpoint,
                suffix=suffix,
                is_best=False,
                is_last=False,
            )
            metrics = self.metrics
            metrics[suffix] = state.valid_metrics
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
        metrics_filename: str = "_metrics.json",
        load_on_stage_end: str = None,
    ):
        """
        Args:
            save_n_best (int): number of best checkpoint to keep,
                if ``0`` then  store only last state of model and
                ``load_on_stage_end`` should be one of
                ``last`` or ``last_full``.
            resume (str): path to checkpoint to load
                and initialize runner state
            resume_dir (str): directory with checkpoints,
                if specified in combination with ``resume``
                than resume checkpoint will be loaded from ``resume_dir``
            metrics_filename (str): filename to save metrics
                in checkpoint folder. Must ends on ``.json`` or ``.yml``
            load_on_stage_end (str): name of the model to load
                at the end of the stage.
                You can use ``best``, ``best_full``
                to load the best model according to validation metrics,
                or ``last`` ``last_full`` (default behaviour)
                to use just the last one.
                If None then no action is required at stage end
                and will be used last state.
        """
        super().__init__(metrics_filename)
        assert load_on_stage_end in [
            None,
            "best",
            "last",
            "best_full",
            "last_full",
        ]
        assert save_n_best >= 0
        if save_n_best == 0:
            assert load_on_stage_end in (None, "last", "last_full")
        if resume_dir is not None:
            assert resume is not None

        self.save_n_best = save_n_best
        self.resume = resume
        self.resume_dir = resume_dir
        self.load_on_stage_end = load_on_stage_end

        self.top_best_metrics = []
        self.metrics_history = []

        self._keys_from_state = ["resume", "resume_dir"]

    def get_checkpoint_suffix(self, checkpoint: dict) -> str:
        """
        Create checkpoint filename suffix based on checkpoint data.

        Args:
            checkpoint (dict): checkpoint dict,
                should contain ``stage_name`` and ``epoch`` keys.
        """
        result = f"{checkpoint['stage_name']}.{checkpoint['epoch']}"
        return result

    def process_metrics(self, last_valid_metrics: Dict[str, float]) -> Dict:
        """
        Add last validation metrics to list of previous validation metrics
        and keep ``save_n_best`` metrics.

        Args:
            last_valid_metrics (dict): dict with metrics
                from last validation step.
        """
        top_best_checkpoints = [
            (Path(filepath).stem, valid_metric)
            for (filepath, _, valid_metric) in self.top_best_metrics
        ]
        all_epochs_metrics = [
            (f"epoch_{order_index}", valid_metric)
            for (order_index, valid_metric) in enumerate(self.metrics_history)
        ]
        metrics = []
        if self.save_n_best > 0:
            best_valid_metrics = top_best_checkpoints[0][1]
            metrics = (
                [("best", best_valid_metrics), ("last", last_valid_metrics)]
                + top_best_checkpoints
                + all_epochs_metrics
            )
        else:
            metrics = [("last", last_valid_metrics)]
        self.metrics = OrderedDict(metrics)
        return self.metrics

    def truncate_checkpoints(self, minimize_metric: bool) -> None:
        """
        Keep ``save_n_best`` checkpoints based on main metric.

        Args:
            minimize_metric (bool): if ``True`` then keep
            ``save_n_best`` checkpoints with the lowest/highest values
            of the main metric.
        """
        self.top_best_metrics = sorted(
            self.top_best_metrics,
            key=lambda x: x[1],
            reverse=not minimize_metric,
        )
        if len(self.top_best_metrics) > self.save_n_best:
            last_item = self.top_best_metrics.pop(-1)
            last_filepath = Path(last_item[0])
            last_filepaths = last_filepath.parent.glob(
                last_filepath.name.replace(".pth", "*")
            )
            for filepath in last_filepaths:
                os.remove(filepath)

    def _save_checkpoint(
        self,
        logdir: Union[str, Path],
        suffix: str,
        checkpoint: Dict,
        is_best: bool,
        is_last: bool,
    ) -> Tuple[str, str]:
        """
        Save checkpoint (simple and full).

        Args:
            logdir (str or Path object): directory for storing checkpoints
            suffix (str): checkpoint suffix
            checkpoint (dict): dict with checkpoint data
            is_best (bool): indicator to save best checkpoint,
                if true then will be saved two additional checkpoints -
                ``best`` and ``best_full``.
            is_last (bool): indicator to save the last checkpoint,
                if true then will be saved two additional checkpoints -
                ``last`` and ``last_full``.
        """
        full_checkpoint_path = utils.save_checkpoint(
            logdir=Path(f"{logdir}/checkpoints/"),
            checkpoint=checkpoint,
            suffix=f"{suffix}_full",
            is_best=is_best,
            is_last=is_last,
            special_suffix="_full",
        )
        exclude = ["criterion", "optimizer", "scheduler"]
        checkpoint_path = utils.save_checkpoint(
            checkpoint={
                key: value
                for key, value in checkpoint.items()
                if all(z not in key for z in exclude)
            },
            logdir=Path(f"{logdir}/checkpoints/"),
            suffix=suffix,
            is_best=is_best,
            is_last=is_last,
        )
        return (full_checkpoint_path, checkpoint_path)

    def process_checkpoint(
        self,
        logdir: Union[str, Path],
        checkpoint: Dict,
        is_best: bool,
        main_metric: str = "loss",
        minimize_metric: bool = True,
    ) -> None:
        """
        Save checkpoint and metrics.

        Args:
            logdir (str or Path object): directory for storing checkpoints
            checkpoint (dict): dict with checkpoint data
            is_best (bool): indicator to save best checkpoint,
                if true then will be saved two additional checkpoints -
                ``best`` and ``best_full``.
            main_metric (str): metric to use for selecting the best model
            minimize_metric (bool): indicator for selecting best metric,
                if true then best metric will be the metric with
                the lowest value, otherwise with the greatest value.
        """
        _, filepath = self._save_checkpoint(
            logdir=logdir,
            checkpoint=checkpoint,
            suffix=self.get_checkpoint_suffix(checkpoint),
            is_best=is_best,
            is_last=True,
        )
        valid_metrics = checkpoint["valid_metrics"]
        checkpoint_metric = valid_metrics[main_metric]
        metrics_record = (filepath, checkpoint_metric, valid_metrics)
        self.top_best_metrics.append(metrics_record)
        self.metrics_history.append(metrics_record)
        self.truncate_checkpoints(minimize_metric=minimize_metric)
        metrics = self.process_metrics(valid_metrics)
        self.save_metric(logdir, metrics)

    def on_stage_start(self, state: State) -> None:
        """
        Setup model for stage.

        NOTE: If CheckpointCallback initialized with ``resume``
        (as path to checkpoint file) or ``resume`` (as filename)
        and ``resume_dir`` (as directory with file)
        then will be performed loading checkpoint.

        Args:
            state (State): training state
        """
        for key in self._keys_from_state:
            value = getattr(state, key, None)
            if value is not None:
                setattr(self, key, value)

        if self.resume_dir is not None:
            self.resume = str(self.resume_dir) + "/" + str(self.resume)

        if self.resume is not None:
            _load_checkpoint(filename=self.resume, state=state)
            self.resume = None

    def on_epoch_end(self, state: State) -> None:
        """
        Collect and save checkpoint after epoch.

        Args:
            state (State): training state
        """
        if state.stage_name.startswith("infer") or state.is_distributed_worker:
            return

        if self.save_n_best > 0:
            checkpoint = _pack_state(state)
            self.process_checkpoint(
                logdir=state.logdir,
                checkpoint=checkpoint,
                is_best=state.is_best_valid,
                main_metric=state.main_metric,
                minimize_metric=state.minimize_metric,
            )

    def on_stage_end(self, state: State) -> None:
        """
        Show information about best checkpoints during the stage and
        load model specified in ``load_on_stage_end``.

        Args:
            state (State): training state
        """
        if state.stage_name.startswith("infer") or state.is_distributed_worker:
            return
        log_message = "Top best models:\n"
        # store latest state
        if self.save_n_best == 0:
            checkpoint = _pack_state(state)
            _, filepath = self._save_checkpoint(
                logdir=state.logdir,
                checkpoint=checkpoint,
                suffix="last",
                is_best=True,  # will duplicate current (last) as best
                is_last=False,  # don't need that because current state is last
            )
            metrics = self.process_metrics(checkpoint["valid_metrics"])
            self.save_metric(state.logdir, metrics)
            main_metric_value = metrics["last"][state.main_metric]
            log_message += "{filepath}\t{metric:3.4f}".format(
                filepath=filepath, metric=main_metric_value
            )
        else:
            log_message += "\n".join(
                [
                    "{filepath}\t{metric:3.4f}".format(
                        filepath=filepath, metric=checkpoint_metric
                    )
                    for filepath, checkpoint_metric, _ in self.top_best_metrics
                ]
            )
        print(log_message)

        if (
            self.load_on_stage_end in ["best", "best_full"]
            and self.save_n_best > 0
        ):
            resume = f"{state.logdir}/checkpoints/{self.load_on_stage_end}.pth"
            print(f"Loading {self.load_on_stage_end} model from {resume}")
            _load_checkpoint(
                filename=resume,
                state=state,
                load_full=self.load_on_stage_end.endswith("full"),
            )


class IterationCheckpointCallback(BaseCheckpointCallback):
    """Iteration checkpoint callback to save your model/criterion/optimizer."""

    def __init__(
        self,
        save_n_last: int = 1,
        period: int = 100,
        stage_restart: bool = True,
        metrics_filename: str = "_metrics_iter.json",
        load_on_stage_end: str = "best_full",
    ):
        """
        Args:
            save_n_last (int): number of last checkpoint to keep
            period (int): save the checkpoint every `period`
            stage_restart (bool): restart counter every stage or not
            metrics_filename (str): filename to save metrics
                in checkpoint folder. Must ends on ``.json`` or ``.yml``
            load_on_stage_end (str): name of the model to load
                at the end of the stage.
                You can use ``best``, ``best_full`` (default)
                to load the best model according to validation metrics,
                or ``last`` ``last_full`` to use just the last one.
        """
        super().__init__(metrics_filename)
        self.save_n_last = save_n_last
        self.period = period
        self.stage_restart = stage_restart
        self._iteration_counter = 0
        self.last_checkpoints = []
        self.metrics_history = []
        self.load_on_stage_end = load_on_stage_end

    def get_checkpoint_suffix(self, checkpoint: dict) -> str:
        """
        Create checkpoint filename suffix based on checkpoint data.

        Args:
            checkpoint (dict): checkpoint dict,
                should contain ``stage_name`` and ``epoch`` keys.
        """
        result = (
            f"{checkpoint['stage_name']}."
            f"epoch.{checkpoint['epoch']}."
            f"iter.{self._iteration_counter}"
        )

        return result

    def process_metrics(self) -> Dict:
        """
        Update metrics with last ``save_n_last`` checkpoints.
        """
        n_last_checkpoints = [
            (Path(filepath).stem, batch_values)
            for (filepath, batch_values) in self.last_checkpoints
        ]
        all_epochs_metrics = [
            (f"epoch_{order_index}", valid_metric)
            for (order_index, valid_metric) in enumerate(self.metrics_history)
        ]

        metrics = OrderedDict(n_last_checkpoints + all_epochs_metrics)
        self.metrics = metrics
        return self.metrics

    def truncate_checkpoints(self, **kwargs) -> None:
        """
        Keep ``save_n_best`` checkpoints based on main metric.
        """
        if len(self.last_checkpoints) > self.save_n_last:
            item = self.last_checkpoints.pop(0)
            top_filepath = item[0]
            os.remove(top_filepath)

    def process_checkpoint(
        self,
        logdir: Union[str, Path],
        checkpoint: Dict,
        batch_metrics: Dict[str, float],
    ):
        """
        Save checkpoint and metrics.

        Args:
            logdir (str or Path object): directory for storing checkpoints
            checkpoint (dict): dict with checkpoint data
            batch_metrics (dict): dict with metrics based on a few batches
        """
        filepath = utils.save_checkpoint(
            logdir=Path(f"{logdir}/checkpoints/"),
            checkpoint=checkpoint,
            suffix=self.get_checkpoint_suffix(checkpoint),
            is_best=False,
            is_last=False,
        )

        self.last_checkpoints.append((filepath, batch_metrics))
        self.truncate_checkpoints()

        self.metrics_history.append(batch_metrics)

        metrics = self.process_metrics()
        self.save_metric(logdir, metrics)
        print(f"\nSaved checkpoint at {filepath}")

    def on_stage_start(self, state: State):
        """
        Reset iterations counter.

        Args:
            state (State): training state
        """
        if self.stage_restart:
            self._iteration_counter = 0

    def on_batch_end(self, state: State):
        """
        Save checkpoint based on batches count.

        Args:
            state (State): training state
        """
        self._iteration_counter += 1
        if self._iteration_counter % self.period == 0:
            checkpoint = _pack_state(state)
            self.process_checkpoint(
                logdir=state.logdir,
                checkpoint=checkpoint,
                batch_metrics=state.batch_metrics,
            )

    def on_stage_end(self, state: State):
        """
        Load model specified in ``load_on_stage_end``.

        Args:
            state (State): training state
        """
        if self.load_on_stage_end in ["best", "best_full"]:
            resume = f"{state.logdir}/checkpoints/{self.load_on_stage_end}.pth"
            print(f"Loading {self.load_on_stage_end} model from {resume}")
            _load_checkpoint(
                filename=resume,
                state=state,
                load_full=self.load_on_stage_end.endswith("full"),
            )


__all__ = ["CheckpointCallback", "IterationCheckpointCallback"]
