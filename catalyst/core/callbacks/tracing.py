from typing import Union
from pathlib import Path
import warnings

from catalyst.core import Callback, CallbackNode, CallbackOrder, IRunner
from catalyst.dl.utils import save_traced_model, trace_model_from_runner


class TracerCallback(Callback):
    """
    Traces model during training if `metric` provided is improved.
    """

    def __init__(
        self,
        metric: str = "loss",
        minimize: bool = True,
        min_delta: float = 1e-6,
        mode: str = "best",
        do_once: bool = True,
        method_name: str = "forward",
        requires_grad: bool = False,
        opt_level: str = None,
        trace_mode: str = "eval",
        out_dir: Union[str, Path] = None,
        out_model: Union[str, Path] = None,
    ):
        """
        Args:
            metric (str): Metric key we should trace model based on
            minimize (bool): Whether do we minimize metric or not
            min_delta (float): Minimum value of change for metric to be
                considered as improved
            mode (str): One of `best` or `last`
            do_once (str): Whether do we trace once per stage or every epoch
            method_name (str): Model's method name that will be
                used as entrypoint during tracing
            requires_grad (bool): Flag to use grads
            opt_level (str): AMP FP16 init level
            trace_mode (str): Mode for model to trace
                (``train`` or ``eval``)
            out_dir (Union[str, Path]): Directory to save model to
            out_model (Union[str, Path]): Path to save model to
                (overrides `out_dir` argument)
        """
        super().__init__(order=CallbackOrder.external, node=CallbackNode.all)

        if trace_mode not in ["train", "eval"]:
            raise ValueError(
                f"Unknown `trace_mode` '{trace_mode}'. "
                f"Must be 'eval' or 'train'"
            )

        if mode not in ["best", "last"]:
            raise ValueError(
                f"Unknown `mode` '{mode}'. " f"Must be 'best' or 'last'"
            )

        if opt_level is not None:
            warnings.warn(
                "TracerCallback: "
                "`opt_level` is not supported yet, "
                "model will be traced as is",
                stacklevel=2,
            )

        self.metric = metric
        self.mode = mode
        self.do_once = do_once
        self.best_score = None
        self.is_better = None
        if minimize:
            self.is_better = lambda score, best: score <= (best - min_delta)
        else:
            self.is_better = lambda score, best: score >= (best + min_delta)

        self.requires_grad = requires_grad
        self.method_name = method_name
        self.trace_mode = trace_mode
        self.opt_level = None

        if out_model is not None:
            out_model = Path(out_model)
        self.out_model = out_model

        if out_dir is not None:
            out_dir = Path(out_dir)
        self.out_dir = out_dir

    def _trace(self, runner: IRunner):
        """
        Performing model tracing on epoch end if condition metric is improved.

        Args:
            runner (IRunner): Current runner
        """
        if self.opt_level is not None:
            device = "cuda"
        else:
            device = "cpu"

        # the only case we need to restore model from previous checkpoint
        # is when we need to trace best model only once in the end of stage
        checkpoint_name_to_restore = None
        if self.do_once and self.mode == "best":
            checkpoint_name_to_restore = "best"

        traced_model = trace_model_from_runner(
            runner=runner,
            checkpoint_name=checkpoint_name_to_restore,
            method_name=self.method_name,
            mode=self.trace_mode,
            requires_grad=self.requires_grad,
            opt_level=self.opt_level,
            device=device,
        )

        save_traced_model(
            model=traced_model,
            logdir=runner.logdir,
            checkpoint_name=self.mode,
            method_name=self.method_name,
            mode=self.trace_mode,
            requires_grad=self.requires_grad,
            opt_level=self.opt_level,
            out_model=self.out_model,
            out_dir=self.out_dir,
        )

    def on_epoch_end(self, runner: IRunner):
        """
        Performing model tracing on epoch end if condition metric is improved

        Args:
            runner (IRunner): Current runner
        """
        if not self.do_once:
            if self.mode == "best":
                score = runner.valid_metrics[self.metric]

                if self.best_score is None:
                    self.best_score = score

                # TODO: EarlyStoppingCallback and TracerCallback
                #  will never work very first epoch
                if self.is_better(score, self.best_score):
                    self.best_score = score
                    self._trace(runner)
            else:
                self._trace(runner)

    def on_stage_end(self, runner: IRunner):
        """
        Performing model tracing on stage end if `do_once` is True.

        Args:
            runner (IRunner): Current runner
        """
        if self.do_once:
            self._trace(runner)


__all__ = ["TracerCallback"]
