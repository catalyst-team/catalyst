from typing import Callable, Dict, Tuple, Union
from collections import OrderedDict
import os
from pathlib import Path

from catalyst.core import utils
from catalyst.core.callback import Callback, CallbackNode, CallbackOrder
from catalyst.core.runner import IRunner


def _pack_runner(runner: IRunner):
    checkpoint = utils.pack_checkpoint(
        model=runner.model,
        criterion=runner.criterion,
        optimizer=runner.optimizer,
        scheduler=runner.scheduler,
        epoch_metrics=dict(runner.epoch_metrics),
        valid_metrics=dict(runner.valid_metrics),
        stage_name=runner.stage_name,
        epoch=runner.epoch,
        loader_name=runner.loader_name,
        loader_step=runner.loader_batch_step,
        global_epoch=runner.global_epoch,
        checkpoint_data=runner.checkpoint_data,
        main_metric=runner.main_metric,
        minimize_metric=runner.minimize_metric,
        valid_loader=runner.valid_loader,
    )
    return checkpoint


def _load_checkpoint(
    *, filename, runner: IRunner, load_full: bool = True
) -> None:
    """
    Load checkpoint from a file.

    Arguments:
        filename (str): path to checkpoint
        runner (IRunner): current runner
        load_full (bool): if true (default) then will be performed
            loading states for criterion, optimizer and scheduler.
            File should contain keys required for
            loading model (``'model_state_dict'``),
            criterion (``'criterion_state_dict'``) (only for full load),
            optimizer (``'optimizer_state_dict'``),
            scheduler (``'scheduler_state_dict'``).

    Raises:
        FileNotFoundError: when file specified in ``filename``
            is not exist.
    """
    if not os.path.isfile(filename):
        raise FileNotFoundError(f"No checkpoint found at {filename}!")

    print(f"=> Loading checkpoint {filename}")
    checkpoint = utils.load_checkpoint(filename)

    if not runner.stage_name.startswith("infer") and load_full:
        runner.stage_name = checkpoint["stage_name"]
        runner.epoch = checkpoint["epoch"]
        runner.global_epoch = checkpoint["global_epoch"]
        # @TODO: should we also load,
        # checkpoint_data, main_metric, minimize_metric, valid_loader ?
        # epoch_metrics, valid_metrics ?

    if load_full:
        utils.unpack_checkpoint(
            checkpoint,
            model=runner.model,
            criterion=runner.criterion,
            optimizer=runner.optimizer,
            scheduler=runner.scheduler,
        )

        print(
            f"loaded state checkpoint {filename} "
            f"(global epoch {checkpoint['global_epoch']}, "
            f"epoch {checkpoint['epoch']}, "
            f"stage {checkpoint['stage_name']})"
        )
    else:
        utils.unpack_checkpoint(
            checkpoint, model=runner.model,
        )

        print(f"loaded model checkpoint {filename}")


def _required_files(logdir: str, load_map: Dict[str, str]) -> Dict[str, str]:
    """
    Generate required files for load model, criterion,
    scheduler, optimizer specified in ``load_map``.

    Expected that ``load_map`` contains keys:
    ``"model"``, ``"criterion"``, ``"optimizer"``, ``"scheduler"``.
    Otherwise an empty dict will be generated.

    Arguments:
        logdir (str): directory with logs
        load_map (Dict[str, str]): dict with specification
            what should be loaded

    Returns:
        Mapping from file to parts required from this file.
    """
    if load_map is None:
        return OrderedDict()

    default_states = {"best", "best_full", "last", "last_full"}
    required_full_checkpoint = ["criterion", "optimizer", "scheduler"]
    experiment_parts = ["model"] + required_full_checkpoint

    # keep required parts
    experiment_parts = list(
        filter(lambda part: part in load_map, experiment_parts)
    )

    # avoid unnecessary loading
    if "model" in experiment_parts and len(experiment_parts) > 1:
        required_full_checkpoint.append("model")

    # mapping - <filename>: <list of parts to load from this file>
    required_files = OrderedDict()
    for part in experiment_parts:
        fname = load_map[part]
        required_full = fname.endswith("_full")
        # specified default state
        if fname in default_states:
            if part in required_full_checkpoint and not required_full:
                fname = fname + "_full"
            fname = f"{logdir}/checkpoints/{fname}.pth"
        # in other case specified path to checkpoint
        required_files[fname] = required_files.get(fname, []) + [part]
    return required_files


def _load_states_from_file_map(
    *, runner: IRunner, load_map: Dict[str, str]
) -> None:
    """
    Load state of a model, criterion, optimizer, scheduler
    from files specified in ``load_map``.

    Arguments:
        runner (IRunner): current runner
        load_map (Dict[str, str]): dict with mappings to load.
            Expected keys - ``'model'``, ``'criterion'``
            ``'optimizer'``, ``'scheduler'``, other keys will be
            ignored.
            Expected that values will be states (``'best'``,
            ``"best_full"``, ``"last"``, ``"last_full"``) or
            path to checkpoint.
            **NOTE:** for successful load criterion, optimizer,
            scheduler states required a full checkpoint.

    Raises:
        FileNotFoundError: when file/state specified in ``load_map``
            is not exist.
    """
    required_files = _required_files(runner.logdir, load_map)

    for filename in required_files.keys():
        if not os.path.isfile(filename):
            raise FileNotFoundError(f"No checkpoint found at {filename}!")

    # extracting parts from files
    for filename, parts_to_load in required_files.items():
        print(f"=> Loading {', '.join(parts_to_load)} from {filename}")
        checkpoint = utils.load_checkpoint(filename)
        to_unpack = {part: getattr(runner, part) for part in parts_to_load}
        utils.unpack_checkpoint(checkpoint, **to_unpack)
        print(f"   loaded: {', '.join(parts_to_load)}")


class ICheckpointCallback(Callback):
    """
    Checkpoint callback interface, abstraction over model checkpointing step.
    """

    pass


class BaseCheckpointCallback(ICheckpointCallback):
    """Base class for all checkpoint callbacks."""

    def __init__(self, metrics_filename: str = "_metrics.json"):
        """
        Args:
            metrics_filename (str): filename to save metrics
                in checkpoint folder. Must ends on ``.json`` or ``.yml``
        """
        super().__init__(
            order=CallbackOrder.external, node=CallbackNode.master
        )
        self.metrics_filename = metrics_filename
        self.metrics: dict = {}

    def _get_checkpoint_suffix(self, checkpoint: dict) -> str:
        return "checkpoint"

    def _save_metric(self, logdir: Union[str, Path], metrics: Dict) -> None:
        utils.save_config(
            metrics, f"{logdir}/checkpoints/{self.metrics_filename}"
        )

    def on_exception(self, runner: IRunner):
        """
        Expection handler.

        Args:
            runner: current runner

        """
        exception = runner.exception
        if not utils.is_exception(exception):
            return

        if runner.device.type == "xla":
            from torch_xla.core.xla_model import save
        else:
            from torch import save

        try:
            checkpoint = _pack_runner(runner)
            suffix = self._get_checkpoint_suffix(checkpoint)
            suffix = f"{suffix}.exception_{exception.__class__.__name__}"
            utils.save_checkpoint(
                logdir=Path(f"{runner.logdir}/checkpoints/"),
                checkpoint=checkpoint,
                suffix=suffix,
                is_best=False,
                is_last=False,
                saver_fn=save,
            )
            metrics = self.metrics
            metrics[suffix] = runner.valid_metrics
            self._save_metric(runner.logdir, metrics)
        except Exception:  # noqa: S110
            pass


class CheckpointCallback(BaseCheckpointCallback):
    """
    Checkpoint callback to save/restore your
    model/criterion/optimizer/scheduler.
    """

    def __init__(
        self,
        save_n_best: int = 1,
        resume: str = None,
        resume_dir: str = None,
        metrics_filename: str = "_metrics.json",
        load_on_stage_start: Union[str, Dict[str, str]] = None,
        load_on_stage_end: Union[str, Dict[str, str]] = None,
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
                in checkpoint folder.
                Must ends on ``.json`` or ``.yml``
            load_on_stage_start (str or Dict[str, str]): load specified
                state/model at stage start.

                If passed **string** then will be performed initialization from
                specified state (``best``/``best_full``/``last``/``last_full``)
                or checkpoint file.

                If passed **dict** then will be performed initialization only
                for specified parts - model, criterion, optimizer, scheduler.

                Example:

                    >>> # possible checkpoints to use:
                    >>> #   "best"/"best_full"/"last"/"last_full"
                    >>> #   or path to specific checkpoint
                    >>> to_load = {
                    >>>    "model": "path/to/checkpoint.pth",
                    >>>    "criterion": "best",
                    >>>    "optimizer": "last_full",
                    >>>    "scheduler": "best_full",
                    >>> }
                    >>> CheckpointCallback(load_on_stage_start=to_load)

                All other keys instead of ``"model"``, ``"criterion"``,
                ``"optimizer"`` and ``"scheduler"`` will be ignored.

                If ``None`` or an empty dict (or dict without mentioned
                above keys) then no action is required at stage start and:

                - Config API - will be used best state of model
                - Notebook API - no action will be performed (will be
                  used the last state)

                **NOTE:** Loading will be performed on all stages except first.

                **NOTE:** Criterion, optimizer and scheduler are optional keys
                and should be loaded from full checkpoint.

                Model state can be loaded from any checkpoint.

                When dict contains keys for model and some other part
                (for example ``{"model": "last", "optimizer": "last"}``)
                and they match in prefix (``"best"`` and
                ``"best_full"``) then will be loaded full checkpoint
                because it contains required states.
            load_on_stage_end (str or Dict[str, str]): load specified
                state/model at stage end.

                If passed **string** then will be performed initialization from
                specified state (``best``/``best_full``/``last``/``last_full``)
                or checkpoint file.

                If passed **dict** then will be performed initialization only
                for specified parts - model, criterion, optimizer, scheduler.
                Logic for dict is the same as for ``load_on_stage_start``.

                If ``None`` then no action is required at stage end
                and will be used the last runner.

                **NOTE:** Loading will be performed always at stage end.
        """
        super().__init__(metrics_filename)
        possible_states = {
            None,
            "best",
            "last",
            "best_full",
            "last_full",
        }
        assert save_n_best >= 0
        if save_n_best == 0:
            assert load_on_stage_end in (None, "last", "last_full")
        if isinstance(load_on_stage_start, str):
            assert load_on_stage_start in possible_states
        if isinstance(load_on_stage_end, str):
            assert load_on_stage_end in possible_states
        if resume_dir is not None:
            assert resume is not None

        self.save_n_best = save_n_best
        self.resume = resume
        self.resume_dir = resume_dir
        self.load_on_stage_start = load_on_stage_start
        self.load_on_stage_end = load_on_stage_end

        self.top_best_metrics = []
        self.metrics_history = []

        self._keys_from_state = ["resume", "resume_dir"]
        self._save_fn: Callable = None

    def _get_checkpoint_suffix(self, checkpoint: dict) -> str:
        """
        Create checkpoint filename suffix based on checkpoint data.

        Args:
            checkpoint (dict): checkpoint dict,
                should contain ``stage_name`` and ``epoch`` keys.

        Returns:
            str: checkpoint suffix
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

        Returns:
            OrderedDict: processed metrics
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
            saver_fn=self._save_fn,
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
            saver_fn=self._save_fn,
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
            suffix=self._get_checkpoint_suffix(checkpoint),
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
        self._save_metric(logdir, metrics)

    @staticmethod
    def _load_runner(
        runner: IRunner,
        mapping: Union[str, Dict[str, str]],
        load_full: bool = False,
    ) -> None:
        """
        Selects a loading method based on type of mapping.

        Args:
            runner (IRunner): current runner
            mapping (str or dict): mapping to use for loading
            load_full (bool): load a full model, used only
                when mapping type is string
        """
        if isinstance(mapping, str):
            if mapping in {"best", "best_full", "last", "last_full"}:
                checkpoint = f"{runner.logdir}/checkpoints/{mapping}.pth"
            else:
                checkpoint = mapping
            _load_checkpoint(
                filename=checkpoint, runner=runner, load_full=load_full,
            )
        elif isinstance(mapping, dict):
            _load_states_from_file_map(
                runner=runner, load_map=mapping,
            )

    def on_stage_start(self, runner: IRunner) -> None:
        """Setup model for stage.

        .. note::

            If CheckpointCallback initialized with
            ``resume`` (as path to checkpoint file)
            or ``resume`` (as filename)
            and ``resume_dir`` (as directory with file)
            then will be performed loading checkpoint.

        Args:
            runner (IRunner): current runner
        """
        if runner.device.type == "xla":
            from torch_xla.core.xla_model import save
        else:
            from torch import save
        self._save_fn = save

        if getattr(runner, "resume", None) is not None:
            self.resume = runner.resume
            runner.resume = None
        elif getattr(runner, "autoresume", None) is not None:
            self.resume_dir = runner.logdir / "checkpoints"
            self.resume = f"{runner.autoresume}_full.pth"
            runner.autoresume = None

        for key in self._keys_from_state:
            value = getattr(runner, key, None)
            if value is not None:
                setattr(self, key, value)

        if self.resume_dir is not None:
            self.resume = str(self.resume_dir) + "/" + str(self.resume)

        if self.resume is not None:
            self._load_runner(runner, mapping=self.resume, load_full=True)
            self.resume = None
        else:
            checkpoint_exists = False
            need_load_full = False
            if isinstance(self.load_on_stage_start, str):
                checkpoint_exists = os.path.isfile(
                    "{}/checkpoints/{}.pth".format(
                        runner.logdir, self.load_on_stage_start
                    )
                )
                need_load_full = self.load_on_stage_start.endswith("full")
            elif isinstance(self.load_on_stage_start, dict):
                required_files = _required_files(
                    runner.logdir, self.load_on_stage_start
                ).keys()
                checkpoint_exists = all(
                    os.path.isfile(file) for file in required_files
                )

            if self.load_on_stage_start is not None and checkpoint_exists:
                self._load_runner(
                    runner,
                    mapping=self.load_on_stage_start,
                    load_full=need_load_full,
                )

    def on_epoch_end(self, runner: IRunner) -> None:
        """
        Collect and save checkpoint after epoch.

        Args:
            runner (IRunner): current runner
        """
        if (
            runner.stage_name.startswith("infer")
            or runner.is_distributed_worker
        ):
            return

        if self.save_n_best > 0:
            checkpoint = _pack_runner(runner)
            self.process_checkpoint(
                logdir=runner.logdir,
                checkpoint=checkpoint,
                is_best=runner.is_best_valid,
                main_metric=runner.main_metric,
                minimize_metric=runner.minimize_metric,
            )

    def on_stage_end(self, runner: IRunner) -> None:
        """
        Show information about best checkpoints during the stage and
        load model specified in ``load_on_stage_end``.

        Args:
            runner (IRunner): current runner
        """
        if (
            runner.stage_name.startswith("infer")
            or runner.is_distributed_worker
        ):
            return
        log_message = "Top best models:\n"
        # store latest state
        if self.save_n_best == 0:
            checkpoint = _pack_runner(runner)
            _, filepath = self._save_checkpoint(
                logdir=runner.logdir,
                checkpoint=checkpoint,
                suffix="last",
                is_best=True,  # will duplicate current (last) as best
                is_last=False,  # don't need that because current state is last
            )
            metrics = self.process_metrics(checkpoint["valid_metrics"])
            self._save_metric(runner.logdir, metrics)
            main_metric_value = metrics["last"][runner.main_metric]
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
        not_required_load_states = {"last", "last_full"}
        if (
            isinstance(self.load_on_stage_end, str)
            and self.load_on_stage_end not in not_required_load_states
            and self.save_n_best > 0
        ):
            need_load_full = (
                self.load_on_stage_end.endswith("full")
                if isinstance(self.load_on_stage_end, str)
                else False
            )
            self._load_runner(
                runner,
                mapping=self.load_on_stage_end,
                load_full=need_load_full,
            )
        elif isinstance(self.load_on_stage_end, dict) and self.save_n_best > 0:
            to_load = {
                k: v
                for k, v in self.load_on_stage_end.items()
                if v not in not_required_load_states
            }
            self._load_runner(runner, mapping=to_load)


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
        self._save_fn = None

    def _get_checkpoint_suffix(self, checkpoint: dict) -> str:
        """
        Create checkpoint filename suffix based on checkpoint data.

        Args:
            checkpoint (dict): checkpoint dict,
                should contain ``stage_name`` and ``epoch`` keys.

        Returns:
            str: checkpoint suffix
        """
        result = (
            f"{checkpoint['stage_name']}."
            f"epoch.{checkpoint['epoch']}."
            f"iter.{self._iteration_counter}"
        )

        return result

    def process_metrics(self) -> Dict:
        """Update metrics with last ``save_n_last`` checkpoints.

        Returns:
            updated metrics
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
        """Keep ``save_n_best`` checkpoints based on main metric.

        Args:
            **kwargs: extra params
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
            suffix=self._get_checkpoint_suffix(checkpoint),
            is_best=False,
            is_last=False,
            saver_fn=self._save_fn,
        )

        self.last_checkpoints.append((filepath, batch_metrics))
        self.truncate_checkpoints()

        self.metrics_history.append(batch_metrics)

        metrics = self.process_metrics()
        self._save_metric(logdir, metrics)
        print(f"\nSaved checkpoint at {filepath}")

    def on_stage_start(self, runner: IRunner):
        """
        Reset iterations counter.

        Args:
            runner (IRunner): current runner
        """
        if self.stage_restart:
            self._iteration_counter = 0

        if runner.device.type == "xla":
            from torch_xla.core.xla_model import save
        else:
            from torch import save
        self._save_fn = save

    def on_batch_end(self, runner: IRunner):
        """
        Save checkpoint based on batches count.

        Args:
            runner (IRunner): current runner
        """
        self._iteration_counter += 1
        if self._iteration_counter % self.period == 0:
            checkpoint = _pack_runner(runner)
            self.process_checkpoint(
                logdir=runner.logdir,
                checkpoint=checkpoint,
                batch_metrics=runner.batch_metrics,
            )

    def on_stage_end(self, runner: IRunner):
        """
        Load model specified in ``load_on_stage_end``.

        Args:
            runner (IRunner): current runner
        """
        if self.load_on_stage_end in ["best", "best_full"]:
            resume = (
                f"{runner.logdir}/checkpoints/{self.load_on_stage_end}.pth"
            )
            print(f"Loading {self.load_on_stage_end} model from {resume}")
            _load_checkpoint(
                filename=resume,
                runner=runner,
                load_full=self.load_on_stage_end.endswith("full"),
            )


__all__ = [
    "CheckpointCallback",
    "IterationCheckpointCallback",
    "ICheckpointCallback",
    "BaseCheckpointCallback",
]
