from typing import Dict
import os

from catalyst.dl.core import Callback, RunnerState
from catalyst.dl import utils


class CheckpointCallback(Callback):
    """
    Checkpoint callback to save/restore your model/criterion/optimizer/metrics.
    """

    def __init__(
        self, save_n_best: int = 3, resume: str = None, resume_dir: str = None
    ):
        """
        Args:
            save_n_best: number of best checkpoint to keep
            resume: path to checkpoint to load and initialize runner state
        """
        self.save_n_best = save_n_best
        self.resume = resume
        self.resume_dir = resume_dir
        self.top_best_metrics = []

        self._keys_from_state = ["resume", "resume_dir"]

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
                f"loaded checkpoint {filename} (epoch {checkpoint['epoch']})")
        else:
            raise Exception("no checkpoint found at {filename}")

    def save_checkpoint(
        self,
        logdir: str,
        checkpoint: Dict,
        is_best: bool,
        save_n_best: int = 5,
        main_metric: str = "loss",
        minimize_metric: bool = True
    ):
        suffix = f"{checkpoint['stage']}.{checkpoint['epoch']}"
        filepath = utils.save_checkpoint(
            logdir=f"{logdir}/checkpoints/",
            checkpoint=checkpoint,
            suffix=suffix,
            is_best=is_best,
            is_last=True
        )

        checkpoint_metric = checkpoint["valid_metrics"][main_metric]
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
        return utils.pack_checkpoint(**kwargs)

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

        checkpoint = self.pack_checkpoint(
            model=state.model,
            criterion=state.criterion,
            optimizer=state.optimizer,
            scheduler=state.scheduler,
            epoch_metrics=dict(state.metrics.epoch_values),
            valid_metrics=dict(state.metrics.valid_values),
            stage=state.stage,
            epoch=state.epoch,
            checkpoint_data=state.checkpoint_data
        )
        self.save_checkpoint(
            logdir=state.logdir,
            checkpoint=checkpoint,
            is_best=state.metrics.is_best,
            save_n_best=self.save_n_best,
            main_metric=state.main_metric,
            minimize_metric=state.minimize_metric
        )

    def on_stage_end(self, state: RunnerState):
        print("Top best models:")
        top_best_metrics_str = "\n".join(
            [
                "{filepath}\t{metric:3.4f}".format(
                    filepath=filepath, metric=metric
                ) for filepath, metric in self.top_best_metrics
            ]
        )
        print(top_best_metrics_str)


class IterationCheckpointCallback(Callback):
    """
    Iteration checkpoint callback to save your model/criterion/optimizer
    """

    def __init__(
        self,
        save_n_last: int = 3,
        num_iters: int = 100,
        stage_restart: bool = True
    ):
        """
        Args:
            save_n_last: number of last checkpoint to keep
            num_iters: save the checkpoint every `num_iters`
            stage_restart: restart counter every stage or not
        """
        self.save_n_last = save_n_last
        self.num_iters = num_iters
        self.stage_restart = stage_restart
        self._iteration_counter = 0
        self.last_checkpoints = []

    def save_checkpoint(
        self,
        logdir,
        checkpoint,
        save_n_last
    ):
        suffix = f"{checkpoint['stage']}." \
                 f"epoch.{checkpoint['epoch']}." \
                 f"iter.{self._iteration_counter}"

        filepath = utils.save_checkpoint(
            logdir=f"{logdir}/checkpoints/",
            checkpoint=checkpoint,
            suffix=suffix,
            is_best=False,
            is_last=False
        )

        self.last_checkpoints.append(filepath)
        if len(self.last_checkpoints) > save_n_last:
            top_filepath = self.last_checkpoints.pop(0)
            os.remove(top_filepath)

        print(f"\nSaved checkpoint at {filepath}")

    def pack_checkpoint(self, **kwargs):
        return utils.pack_checkpoint(**kwargs)

    def on_stage_start(self, state):
        if self.stage_restart:
            self._iteration_counter = 0

    def on_batch_end(self, state):
        self._iteration_counter += 1
        if self._iteration_counter % self.num_iters == 0:
            checkpoint = self.pack_checkpoint(
                model=state.model,
                criterion=state.criterion,
                optimizer=state.optimizer,
                scheduler=state.scheduler,
                epoch_metrics=None,
                valid_metrics=None,
                stage=state.stage,
                epoch=state.epoch
            )
            self.save_checkpoint(
                logdir=state.logdir,
                checkpoint=checkpoint,
                save_n_last=self.save_n_last
            )


__all__ = ["CheckpointCallback", "IterationCheckpointCallback"]
