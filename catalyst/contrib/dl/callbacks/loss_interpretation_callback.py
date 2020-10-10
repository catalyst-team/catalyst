from typing import Any, Callable, Container, Sequence

import numpy as np

import torch
from torch.utils.data import SequentialSampler

from catalyst.contrib.tools import SummaryWriter
from catalyst.core import IMetricCallback, IRunner


def img_publisher(writer: SummaryWriter, tag, sample):
    writer.add_image(f"{tag}_images", sample)


def text_publisher(writer: SummaryWriter, tag, sample):
    writer.add_text(f"{tag}_text", sample)


class LossInterpretationCallback(IMetricCallback):
    def __init__(
        self,
        criterion=None,
        loaders_to_skip: Container[str] = (),
        prefix: str = "",
        input_key: str = "targets",
        output_key: str = "logits",
        idx_key=None,
        top_k=10,
        tensorboard_sequence: Sequence = None,
        tensorboard_publishers: Sequence[
            Callable[[SummaryWriter, str, Any], Any]
        ] = (),
        **loss_kwargs,
    ):
        """
        Serializes per-sample loss values, and logs the best and worst to the tensorboard.
        Args:
            criterion: If the criterion is not available in the runner, this will be used.
            loaders_to_skip: A list of loaders to be skipped.
            prefix: A prefix for the Tensorboard tags.
            input_key: The key for the targets.
            output_key: The key for the predictions.
            idx_key: Optional, if your DataLoader is shuffled, you can optionally add an index key to your dictionary.
            top_k: Number of extreme examples to be logged to the Tensorboard.
            tensorboard_sequence: A sequence that will be used to publish to the Tensorboard.
            tensorboard_publishers: A list of functions that accept the summary writer, the tag and the sample.
            **loss_kwargs: additional args to be passed to the loss.
        """
        super().__init__(prefix, input_key, output_key, **loss_kwargs)
        self.metric = criterion
        self.interpretations = {}
        self.top_k = top_k
        self.loggers = {}
        self.tensorboard_sequence = tensorboard_sequence
        self.tensorboard_publishers = tensorboard_publishers
        self._loaders_to_skip = loaders_to_skip
        self._idx_key = idx_key

    def _should_interpret_loader(self, runner: IRunner):
        if runner.loader_name in self._loaders_to_skip:
            return False
        if isinstance(
            runner.loaders[runner.loader_name].sampler, SequentialSampler
        ):
            return True

        """
        If the sampler is not sequential, we cannot recover the original index of the sample,
        unless the user has provided `idx_key`.
        See: https://github.com/catalyst-team/catalyst/issues/950#issuecomment-703220633
        """
        return self._idx_key is not None

    def on_loader_start(self, runner: IRunner):
        if not self._should_interpret_loader(runner):
            return
        if self.metric is None and runner.criterion is None:
            message = f"""LossInterpretationCallback needs a criterion either
from the runner, or from the "criterion" constructor parameter.
Since neither was provided, skipping loader "{runner.loader_name}".
            """
            print(message)
            return
        self.metric = runner.criterion if self.metric is None else self.metric
        if runner.loader_name not in self.loggers:
            logdir = runner.logdir / f"{runner.loader_name}_log"
            self.loggers[runner.loader_name] = SummaryWriter(str(logdir))
        if runner.loader_name not in self.interpretations:
            self.interpretations[runner.loader_name] = {
                "loss": [],
                "indices": [],
            }

    def on_loader_end(self, runner: IRunner):
        if not self._should_interpret_loader(runner):
            return
        if self.metric is None:
            return

        self.interpretations[runner.loader_name] = {
            key: np.concatenate(value, axis=0)
            for key, value in self.interpretations[runner.loader_name].items()
        }

        out_file = runner.logdir / f"{runner.loader_name}_interpretations.pkl"
        torch.save(self.interpretations[runner.loader_name], out_file)

        loss_sorter = self.interpretations[runner.loader_name][
            "loss"
        ].argsort()
        indices_sorted = self.interpretations[runner.loader_name]["indices"][
            loss_sorter
        ]
        indices = {
            "best": indices_sorted[: self.top_k],
            "worst": indices_sorted[-self.top_k :],
        }

        writer: SummaryWriter = self.loggers[runner.loader_name]
        for type_prefix in ["best", "worst"]:
            for idx in indices[type_prefix]:
                tag = f"{self.prefix}{type_prefix}"
                for tensorboard_publisher in self.tensorboard_publishers:
                    sample = self.tensorboard_sequence[idx]
                    tensorboard_publisher(writer, tag, sample)

    def on_batch_end(self, runner: IRunner):
        if not self._should_interpret_loader(runner):
            return
        if self.metric is None:
            return
        loss_items: torch.Tensor = self._compute_metric_value(
            runner.output, runner.input
        )
        loss_items = loss_items.unsqueeze(-1).unsqueeze(-1)
        if len(loss_items.shape) > 1:
            dims = tuple(range(1, len(loss_items.shape)))
            loss_items = loss_items.mean(dim=dims)

        if self._idx_key is None:
            bs = len(loss_items)
            indices_so_far = self.interpretations[runner.loader_name][
                "indices"
            ]
            start_idx = (
                0 if len(indices_so_far) == 0 else (indices_so_far[-1][-1] + 1)
            )
            indices = np.arange(start_idx, start_idx + bs)
        else:
            indices = runner.input[self._idx_key].detach().cpu().numpy()

        self.interpretations[runner.loader_name]["loss"].append(
            loss_items.detach().cpu().numpy()
        )
        self.interpretations[runner.loader_name]["indices"].append(indices)

    @property
    def metric_fn(self):
        return self.metric
