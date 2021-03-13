from typing import Dict, Union
from collections import OrderedDict
import os
from pathlib import Path
import shutil

import torch.distributed as dist

from catalyst.core.callback import Callback, CallbackNode, CallbackOrder
from catalyst.core.runner import IRunner
from catalyst.tools.metric_handler import MetricHandler
from catalyst.utils.config import save_config


def _save_checkpoint(
    checkpoint: Dict,
    runner: "IRunner",
    logdir: Union[Path, str],
    suffix: str,
    is_best: bool = False,
    is_last: bool = False,
    extra_suffix: str = "",
) -> Union[Path, str]:
    """Saving checkpoint to a file.

    Args:
        checkpoint: data to save.
        runner: current runner
        logdir: directory where checkpoint should be stored.
        suffix: checkpoint file name.
        is_best: if ``True`` then also will be generated best checkpoint file.
        is_last: if ``True`` then also will be generated last checkpoint file.
        extra_suffix: suffix to use for saving best/last checkpoints.

    Returns:
        path to saved checkpoint
    """
    os.makedirs(logdir, exist_ok=True)
    filename = f"{logdir}/{suffix}.pth"
    runner.engine.save_checkpoint(checkpoint, filename)
    if is_best:
        shutil.copyfile(filename, f"{logdir}/best{extra_suffix}.pth")
    if is_last:
        shutil.copyfile(filename, f"{logdir}/last{extra_suffix}.pth")
    return filename


def _load_checkpoint(*, filename, runner: "IRunner", load_full: bool = True) -> None:
    """
    Load checkpoint from a file.

    Arguments:
        filename: path to checkpoint
        runner: current runner
        load_full: if true (default) then will be performed
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
    is_master_process = runner.engine.is_master_process

    if not os.path.isfile(filename):
        raise FileNotFoundError(f"No checkpoint found at {filename}!")

    if is_master_process:
        print(f"=> Loading checkpoint {filename}")
    checkpoint = runner.engine.load_checkpoint(filename)

    if not runner.stage_key.startswith("infer") and load_full:
        runner.global_epoch_step = checkpoint["global_epoch_step"]
        runner.global_batch_step = checkpoint["global_batch_step"]
        runner.global_sample_step = checkpoint["global_sample_step"]

    if load_full:
        runner.engine.unpack_checkpoint(
            checkpoint,
            model=runner.model,
            criterion=runner.criterion,
            optimizer=runner.optimizer,
            scheduler=runner.scheduler,
        )

        if is_master_process:
            print(
                f"full checkpoint {filename} loaded "
                f"(global epoch {checkpoint['global_epoch_step']}, "
                f"stage {checkpoint['stage_key']}, "
                f"epoch {checkpoint['stage_epoch_step']})"
            )
    else:
        runner.engine.unpack_checkpoint(checkpoint, model=runner.model)

        if is_master_process:
            print(
                f"model checkpoint {filename} loaded "
                f"(global epoch {checkpoint['global_epoch_step']}, "
                f"stage {checkpoint['stage_key']}, "
                f"epoch {checkpoint['stage_epoch_step']})"
            )


def _get_required_files(logdir: str, load_map: Dict[str, str]) -> Dict[str, str]:
    """
    Generate required files for load model, criterion,
    scheduler, optimizer specified in ``load_map``.

    Expected that ``load_map`` contains keys:
    ``"model"``, ``"criterion"``, ``"optimizer"``, ``"scheduler"``.
    Otherwise an empty dict will be generated.

    Arguments:
        logdir: directory with logs
        load_map (Dict[str, str]): dict with specification what should be loaded

    Returns:
        Mapping from file to parts required from this file.
    """
    if load_map is None:
        return OrderedDict()

    default_states = {"best", "best_full", "last", "last_full"}
    required_full_checkpoint = ["criterion", "optimizer", "scheduler"]
    steps = ["global_epoch_step", "global_batch_step", "global_sample_step"]
    experiment_parts = ["model"] + required_full_checkpoint + steps

    # keep required parts
    experiment_parts = list(filter(lambda part: part in load_map, experiment_parts))

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
            fname = f"{logdir}/{fname}.pth"
        # in other case specified path to checkpoint
        required_files[fname] = required_files.get(fname, []) + [part]
    return required_files


def _load_states_from_file_map(
    *, logdir: str, runner: "IRunner", load_map: Dict[str, str]
) -> None:
    """
    Load state of a model, criterion, optimizer, scheduler
    from files specified in ``load_map``.

    Arguments:
        runner: current runner
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
    required_files = _get_required_files(logdir, load_map)

    for filename in required_files.keys():
        if not os.path.isfile(filename):
            raise FileNotFoundError(f"No checkpoint found at {filename}!")

    # extracting parts from files
    for filename, parts_to_load in required_files.items():
        print(f"=> Loading {', '.join(parts_to_load)} from {filename}")
        checkpoint = runner.engine.load_checkpoint(filename)
        to_unpack = {part: getattr(runner, part) for part in parts_to_load}
        runner.engine.unpack_checkpoint(checkpoint, **to_unpack)
        # hotfix
        if "global_epoch_step" in to_unpack:
            runner.global_epoch_step = checkpoint["global_epoch_step"]
        if "global_batch_step" in to_unpack:
            runner.global_batch_step = checkpoint["global_batch_step"]
        if "global_sample_step" in to_unpack:
            runner.global_sample_step = checkpoint["global_sample_step"]
        print(f"   loaded: {', '.join(parts_to_load)}")


def _load_runner(
    logdir: str, runner: "IRunner", mapping: Union[str, Dict[str, str]], load_full: bool = False,
) -> None:
    """
    Selects a loading method based on type of mapping.

    Args:
        logdir: logdir with checkpoints
        runner: current runner
        mapping: mapping to use for loading
        load_full: load a full model, used only when mapping type is string
    """
    if isinstance(mapping, str):
        if mapping in {"best", "best_full", "last", "last_full"}:
            checkpoint = f"{logdir}/{mapping}.pth"
        else:
            checkpoint = mapping
        _load_checkpoint(filename=checkpoint, runner=runner, load_full=load_full)
    elif isinstance(mapping, dict):
        _load_states_from_file_map(logdir=logdir, runner=runner, load_map=mapping)


class ICheckpointCallback(Callback):
    """Criterion callback interface, abstraction over checkpoint step."""

    pass


class CheckpointCallback(ICheckpointCallback):
    """Checkpoint callback to save/restore your model/criterion/optimizer/scheduler.

    Args:
        logdir: directory to store chekpoints
        loader_key: loader key for best model selection (based on metric score over the dataset)
        metric_key: metric key for best model selection (based on metric score over the dataset)
        minimize: boolean flag to minimize the required metric
        min_delta: minimal delta for metric improve
        save_n_best: number of best checkpoint to keep,
            if ``0`` then  store only last state of model and
            ``load_on_stage_end`` should be one of
            ``last`` or ``last_full``.
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
            - Notebook API - no action will be performed (will be used the last state)

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
        metrics_filename: filename to save metrics
            in checkpoint folder.
            Must ends on ``.json`` or ``.yml``
        mode: checkpoining mode, could be ``all``, ``full``, ``model``
        use_logdir_postfix: boolean flag to use extra prefix ``checkpoints`` for logdir
        use_runner_logdir: boolean flag to use ``runner._logdir`` as logdir
    """

    def __init__(
        self,
        logdir: str = None,
        # model selection info
        loader_key: str = None,
        metric_key: str = None,
        minimize: bool = None,
        min_delta: float = 1e-6,
        save_n_best: int = 1,
        # loading info
        load_on_stage_start: Union[str, Dict[str, str]] = None,
        load_on_stage_end: Union[str, Dict[str, str]] = None,
        # resume: str = None,
        # resume_dir: str = None,
        # checkpointer info
        metrics_filename: str = "_metrics.json",
        mode: str = "all",
        use_logdir_postfix: bool = False,
        use_runner_logdir: bool = False,
    ):
        """Init."""
        super().__init__(order=CallbackOrder.external, node=CallbackNode.all)
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
        # if resume_dir is not None:
        #     assert resume is not None

        if loader_key is not None or metric_key is not None:
            assert loader_key is not None and metric_key is not None, (
                "For checkpoint selection `CheckpointCallback` "
                "requires both `loader_key` and `metric_key` specified."
            )
            self._use_model_selection = True
            self.minimize = minimize if minimize is not None else True  # loss-oriented selection
        else:
            self._use_model_selection = False
            self.minimize = False  # epoch-num-oriented selection

        assert mode in (
            "all",
            "full",
            "model",
        ), "`CheckpointCallback` could work only in `all`, `full` or `model` modes."

        # checkpointer info
        self.logdir = logdir
        self.mode = mode
        self.metrics_filename = metrics_filename
        self.use_logdir_postfix = use_logdir_postfix
        self.use_runner_logdir = use_runner_logdir
        assert (
            self.logdir is not None or self.use_runner_logdir
        ), "CheckpointCallback requires specified `logdir`"

        # model selection info
        self.loader_key = loader_key
        self.metric_key = metric_key
        self.is_better = MetricHandler(minimize=minimize, min_delta=min_delta)
        self.save_n_best = save_n_best
        # list with topN metrics [(score, filepath, stage_key, stage_epoch_step, epoch metrics)]
        self.top_best_metrics = []
        self.best_score = None

        # loading info
        self.load_on_stage_start = load_on_stage_start
        self.load_on_stage_end = load_on_stage_end
        # self.resume = resume
        # self.resume_dir = resume_dir

    def _pack_checkpoint(self, runner: "IRunner"):
        checkpoint = runner.engine.pack_checkpoint(
            model=runner.model,
            criterion=runner.criterion,
            optimizer=runner.optimizer,
            scheduler=runner.scheduler,
            # experiment info
            run_key=runner.run_key,
            global_epoch_step=runner.global_epoch_step,
            global_batch_step=runner.global_batch_step,
            global_sample_step=runner.global_sample_step,
            # stage info
            stage_key=runner.stage_key,
            stage_epoch_step=runner.stage_epoch_step,
            stage_batch_step=runner.stage_batch_step,
            stage_sample_step=runner.stage_sample_step,
            # epoch info
            epoch_metrics={k: dict(v) for k, v in runner.epoch_metrics.items()},
            # loader info
            loader_key=runner.loader_key,
            loader_batch_step=runner.loader_batch_step,
            loader_sample_step=runner.loader_sample_step,
            # checkpointer info
            checkpointer_loader_key=self.loader_key,
            checkpointer_metric_key=self.metric_key,
            checkpointer_minimize=self.minimize,
        )
        return checkpoint

    def _save_checkpoint(
        self, runner: IRunner, checkpoint: Dict, is_best: bool, is_last: bool
    ) -> str:
        """
        Saves checkpoints: full with model/criterion/optimizer/scheduler
        and truncated with model only.
        """
        logdir = Path(f"{self.logdir}/")
        suffix = f"{runner.stage_key}.{runner.stage_epoch_step}"
        checkpoint_path = None

        if self.mode in ("all", "full"):
            checkpoint_path = _save_checkpoint(
                runner=runner,
                logdir=logdir,
                checkpoint=checkpoint,
                suffix=f"{suffix}_full",
                is_best=is_best,
                is_last=is_last,
                extra_suffix="_full",
            )
        if self.mode in ("all", "model"):
            exclude = ["criterion", "optimizer", "scheduler"]
            checkpoint_path = _save_checkpoint(
                runner=runner,
                checkpoint={
                    key: value
                    for key, value in checkpoint.items()
                    if all(z not in key for z in exclude)
                },
                logdir=logdir,
                suffix=suffix,
                is_best=is_best,
                is_last=is_last,
            )
        return checkpoint_path

    def _truncate_checkpoints(self) -> None:
        self.top_best_metrics = sorted(
            self.top_best_metrics, key=lambda x: x[0], reverse=not self.minimize,
        )
        if len(self.top_best_metrics) > self.save_n_best:
            last_item = self.top_best_metrics.pop(-1)
            last_filepath = Path(last_item[1])
            last_filepaths = last_filepath.parent.glob(last_filepath.name.replace(".pth", "*"))
            for filepath in last_filepaths:
                os.remove(filepath)

    def _prepare_metrics_log(self, last_epoch_score: float, last_epoch_metrics: Dict) -> Dict:
        top_best_checkpoints = [
            (Path(filepath).stem, {**epoch_metrics, **{"_score_": score}})
            for (score, filepath, _, _, epoch_metrics) in self.top_best_metrics
        ]
        if self.save_n_best > 0:
            best_epoch_score = top_best_checkpoints[0][0]
            best_epoch_metrics = top_best_checkpoints[0][-1]
            metrics = [
                ("best", {**best_epoch_metrics, **{"_score_": best_epoch_score}}),
                ("last", {**last_epoch_metrics, **{"_score_": last_epoch_score}}),
            ] + top_best_checkpoints
        else:
            metrics = [("last", {**last_epoch_metrics, **{"_score_": last_epoch_score}})]
        return OrderedDict(metrics)

    def on_stage_start(self, runner: "IRunner") -> None:
        """Setup model for stage.

        .. note::

            If CheckpointCallback initialized with
            ``resume`` (as path to checkpoint file)
            or ``resume`` (as filename)
            and ``resume_dir`` (as directory with file)
            then will be performed loading checkpoint.

        Raises:
            FileNotFoundError: if specified load_on_stage_start
                but checkpoint file is missing.

        Args:
            runner: current runner
        """
        if runner.is_infer_stage:
            return
        # @TODO: very tricky hack, should be removed
        if self.logdir is None and self.use_runner_logdir:
            self.logdir = getattr(runner, "_logdir", None)
            if self.use_logdir_postfix:
                self.logdir = os.path.join(self.logdir, "checkpoints")

        # @TODO:
        # # Use a barrier() to make sure that process 1 loads the model after process 0 saves it.
        # dist.barrier()
        # # configure map_location properly
        # map_location = {"cuda:%d" % 0: "cuda:%d" % rank}
        # ddp_model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=map_location))
        # Use a barrier() to make sure that all processes have finished reading the checkpoint
        # dist.barrier()

        is_first_stage = list(runner.stages).index(runner.stage_key) == 0
        if self.load_on_stage_start is not None and not is_first_stage:
            need_full = False
            file_exists = False
            if isinstance(self.load_on_stage_start, str):
                need_full = self.load_on_stage_start.endswith("full")
                use_file = os.path.join(self.logdir, f"{self.load_on_stage_start}.pth")
                file_exists = os.path.isfile(use_file)
                if not file_exists:
                    raise FileNotFoundError(f"Missing file '{use_file}'!")  # noqa: F821
            elif isinstance(self.load_on_stage_start, dict):
                required_files = _get_required_files(self.logdir, self.load_on_stage_start).keys()
                file_exists = True
                for use_file in required_files:
                    if not os.path.isfile(use_file):
                        file_exists = False
                        raise FileNotFoundError(f"Missing file '{use_file}'!")

            if self.load_on_stage_start is not None and file_exists:
                _load_runner(
                    logdir=self.logdir,
                    runner=runner,
                    mapping=self.load_on_stage_start,
                    load_full=need_full,
                )

    #     if getattr(runner, "resume", None) is not None:
    #         self.resume = runner.resume
    #         runner.resume = None
    #     elif getattr(runner, "autoresume", None) is not None:
    #         self.resume_dir = runner.logdir / "checkpoints"
    #         self.resume = f"{runner.autoresume}_full.pth"
    #         runner.autoresume = None
    #
    #     for key in self._keys_from_runner:
    #         value = getattr(runner, key, None)
    #         if value is not None:
    #             setattr(self, key, value)
    #
    #     if self.resume_dir is not None:
    #         self.resume = str(self.resume_dir) + "/" + str(self.resume)
    #
    #     if self.resume is not None:
    #         _load_runner(logdir=self.logdir, runner=runner, mapping=self.resume, load_full=True)
    #         self.resume = None
    #     else:
    #         checkpoint_exists = False
    #         need_load_full = False
    #         if isinstance(self.load_on_stage_start, str):
    #             checkpoint_exists =
    #               os.path.isfile(f"{self.logdir}/{self.load_on_stage_start}.pth")
    #             need_load_full = self.load_on_stage_start.endswith("full")
    #         elif isinstance(self.load_on_stage_start, dict):
    #             required_files =
    #               _get_required_files(self.logdir, self.load_on_stage_start).keys()
    #             checkpoint_exists = all(os.path.isfile(file) for file in required_files)
    #
    #         if self.load_on_stage_start is not None and checkpoint_exists:
    #             _load_runner(
    #                 logdir=self.logdir,
    #                 runner=runner,
    #                 mapping=self.load_on_stage_start,
    #                 load_full=need_load_full,
    #             )

    def on_epoch_end(self, runner: "IRunner") -> None:
        """
        Collects and saves checkpoint after epoch.

        Args:
            runner: current runner
        """
        if runner.is_infer_stage:
            return
        if runner.engine.is_ddp and not runner.engine.is_master_process:
            return

        if self._use_model_selection:
            # score model based on the specified metric
            score = runner.epoch_metrics[self.loader_key][self.metric_key]
        else:
            # score model based on epoch number
            score = runner.global_epoch_step

        is_best = False
        if self.best_score is None or self.is_better(score, self.best_score):
            self.best_score = score
            is_best = True

        if self.save_n_best > 0:
            # pack checkpoint
            checkpoint = self._pack_checkpoint(runner)
            # save checkpoint
            checkpoint_path = self._save_checkpoint(
                runner=runner, checkpoint=checkpoint, is_best=is_best, is_last=True,
            )
            # add metrics to records
            metrics_record = (
                float(score),
                checkpoint_path,
                runner.stage_key,
                runner.stage_epoch_step,
                dict(runner.epoch_metrics),
            )
            self.top_best_metrics.append(metrics_record)
            # truncate checkpoints
            self._truncate_checkpoints()
            # save checkpoint metrics
            metrics_log = self._prepare_metrics_log(float(score), dict(runner.epoch_metrics))
            save_config(metrics_log, f"{self.logdir}/{self.metrics_filename}")

    def on_stage_end(self, runner: "IRunner") -> None:
        """
        Show information about best checkpoints during the stage and
        load model specified in ``load_on_stage_end``.

        Args:
            runner: current runner
        """
        if runner.is_infer_stage:
            return
        if runner.engine.is_ddp and not runner.engine.is_master_process:
            # worker sync
            dist.barrier()
            return

        # let's log Top-N base metrics
        log_message = "Top best models:\n"
        # store latest state
        if self.save_n_best == 0:
            score = runner.epoch_metrics[self.loader_key][self.metric_key]
            # pack checkpoint
            checkpoint = self._pack_checkpoint(runner)
            # save checkpoint
            checkpoint_path = self._save_checkpoint(
                runner=runner,
                checkpoint=checkpoint,
                is_best=True,  # will duplicate current (last) as best
                is_last=False,  # don't need that because current state is last
            )
            # add metrics to records
            # save checkpoint metrics
            metrics_log = self._prepare_metrics_log(float(score), dict(runner.epoch_metrics))
            save_config(metrics_log, f"{self.logdir}/{self.metrics_filename}")
            log_message += f"{checkpoint_path}\t{score:3.4f}"
        else:
            log_message += "\n".join(
                [f"{filepath}\t{score:3.4f}" for score, filepath, _, _, _ in self.top_best_metrics]
            )
        print(log_message)

        # let's load runner state (model, criterion, optimizer, scheduler) if required
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
            _load_runner(
                logdir=self.logdir,
                runner=runner,
                mapping=self.load_on_stage_end,
                load_full=need_load_full,
            )
        elif isinstance(self.load_on_stage_end, dict) and self.save_n_best > 0:
            to_load = {
                k: v
                for k, v in self.load_on_stage_end.items()
                if v not in not_required_load_states
            }
            _load_runner(logdir=self.logdir, runner=runner, mapping=to_load)

        if runner.engine.is_ddp and runner.engine.is_master_process:
            # master sync
            dist.barrier()


__all__ = ["ICheckpointCallback", "CheckpointCallback"]
