from typing import Union, List, Any, Dict, Callable

import torch
import torchvision.utils

from catalyst.dl import RunnerState
from catalyst.dl import registry
from catalyst.dl.callbacks import CriterionAggregatorCallback, OptimizerCallback, CriterionCallback
from catalyst.dl.core import Callback, CallbackOrder, MetricCallback
from catalyst.utils.tensorboard import SummaryWriter

# TODO: implement abstract class and move to core catalyst
# @registry.Callback
# class MultiInputOutputMetric(MetricCallback):
#     """Class similar to MetricCallback,
#     but it can accept arbitrary number of input and output keys for main criterion function"""
#
#     def __init__(self, prefix: str, metric_fn: Callable, input_key: str = "targets", output_key: str = "logits",
#                  **metric_params):
#         super().__init__(prefix, metric_fn, input_key, output_key, **metric_params)
#
#     def on_batch_end(self, state: RunnerState):
#         super().on_batch_end(state)


@registry.Callback
class CriterionWithAdditionalArgsCallback(CriterionCallback):
    """TODO: merge with CriterionCallback
    """

    def _get_additional_criterion_args(self, state: RunnerState):
        return {}

    def _compute_loss(self, state: RunnerState, criterion):
        output = self._get(state.output, self.output_key)
        input = self._get(state.input, self.input_key)

        kwargs = self._get_additional_criterion_args(state)
        loss = criterion(output, input, **kwargs)
        return loss


@registry.Callback
class CriterionWithDiscriminatorCallback(CriterionWithAdditionalArgsCallback):
    """TODO: docs"""

    def __init__(self, input_key: Union[str, List[str]] = "targets", output_key: Union[str, List[str]] = "logits",
                 prefix: str = "loss", criterion_key: str = None, multiplier: float = 1.0,
                 discriminator_model_key="discriminator", discriminator_model_criterion_key="discriminator"):
        super().__init__(input_key, output_key, prefix, criterion_key, multiplier)
        self.discriminator_model_key = discriminator_model_key
        self.discriminator_model_criterion_key = discriminator_model_criterion_key

    def _get_additional_criterion_args(self, state: RunnerState):
        return {self.discriminator_model_criterion_key: state.model[self.discriminator_model_key]}


@registry.Callback
class LipzOptimizerCallback(OptimizerCallback):
    """
    Optimizer callback + weights clipping after step is finished
    """

    def __init__(
            self,
            grad_clip_params: Dict = None,
            accumulation_steps: int = 1,
            optimizer_key: str = None,
            loss_key: str = "loss",
            decouple_weight_decay: bool = True,
            weight_clamp_value: float = 0.1
    ):
        super().__init__(
            grad_clip_params=grad_clip_params,
            accumulation_steps=accumulation_steps,
            optimizer_key=optimizer_key,
            loss_key=loss_key,
            decouple_weight_decay=decouple_weight_decay
        )
        self.weight_clamp_value = weight_clamp_value

    def on_batch_end(self, state):
        """On batch end event"""
        super().on_batch_end(state)
        if not state.need_backward:
            return

        optimizer = state.get_key(
            key="optimizer", inner_key=self.optimizer_key
        )

        need_gradient_step = \
            self._accumulation_counter % self.accumulation_steps == 0

        if need_gradient_step:
            for group in optimizer.param_groups:
                for param in group["params"]:
                    param.data.clamp_(min=-self.weight_clamp_value, max=self.weight_clamp_value)


@registry.Callback
class WeightedCriterionAggregatorCallback(CriterionAggregatorCallback):
    """
    Weighted criterion aggregation
    """

    def __init__(self, prefix: str, loss_keys: Union[str, List[str]] = None,
                 loss_aggregate_fn: str = "mean",
                 weights: List[float] = None,
                 multiplier: float = 1.0) -> None:
        super().__init__(prefix, loss_keys, loss_aggregate_fn="sum", multiplier=multiplier)
        # note that we passed `loss_aggregate_fn="sum"` always to reuse parent's `on_batch_end` method unchanged
        # we use "sum" as after `_preprocess_loss` individual losses are already weighted and we need to sum them

        assert self.loss_keys is not None
        assert weights is not None and len(weights) == len(self.loss_keys)
        self.weights = weights
        if loss_aggregate_fn == "mean":
            self.weights = [w / sum(weights) for w in weights]

    def _preprocess_loss(self, loss: Any) -> List[torch.Tensor]:
        assert isinstance(loss, dict)
        return [loss[key] * self.weights[i] for i, key in enumerate(self.loss_keys)]


@registry.Callback
class VisualizationCallback(Callback):
    TENSORBOARD_LOGGER_KEY = "tensorboard"

    def __init__(
            self,
            input_keys=None,
            output_keys=None,
            batch_frequency=25,
            concat_images=True,
            max_images=20,
            n_row=1,
            denorm="default"
    ):
        super().__init__(CallbackOrder.Other)
        if input_keys is None:
            self.input_keys = []
        elif isinstance(input_keys, str):
            self.input_keys = [input_keys]
        elif isinstance(input_keys, (tuple, list)):
            assert all(isinstance(k, str) for k in input_keys)
            self.input_keys = list(input_keys)
        else:
            raise ValueError(
                f"Unexpected format of 'input_keys' "
                f"argument: must be string or list/tuple"
            )

        if output_keys is None:
            self.output_keys = []
        elif isinstance(output_keys, str):
            self.output_keys = [output_keys]
        elif isinstance(output_keys, (tuple, list)):
            assert all(isinstance(k, str) for k in output_keys)
            self.output_keys = list(output_keys)
        else:
            raise ValueError(
                f"Unexpected format of 'output_keys' "
                f"argument: must be string or list/tuple"
            )

        if len(self.input_keys) + len(self.output_keys) == 0:
            raise ValueError("Useless visualizer: pass at least one image key")

        self.batch_frequency = int(batch_frequency)
        assert self.batch_frequency > 0

        self.concat_images = concat_images
        self.max_images = max_images
        if denorm.lower() == "default":
            # normalization from [-1, 1] to [0, 1] (the latter is valid for tb)
            self.denorm = lambda x: x / 2 + .5
        elif denorm is None or denorm.lower() == "none":
            self.denorm = lambda x: x
        else:
            raise ValueError("unknown denorm fn")
        self._n_row = n_row
        self._reset()

    def _reset(self):
        self._loader_batch_count = 0
        self._loader_visualized_in_current_epoch = False

    @staticmethod
    def _get_tensorboard_logger(state: RunnerState) -> SummaryWriter:
        tb_key = VisualizationCallback.TENSORBOARD_LOGGER_KEY
        if (
                tb_key in state.loggers
                and state.loader_name in state.loggers[tb_key].loggers
        ):
            return state.loggers[tb_key].loggers[state.loader_name]
        raise RuntimeError(
            f"Cannot find Tensorboard logger for loader {state.loader_name}"
        )

    def compute_visualizations(self, state):
        input_tensors = [
            state.input[input_key] for input_key in self.input_keys
        ]
        output_tensors = [
            state.output[output_key] for output_key in self.output_keys
        ]
        visualizations = dict()
        if self.concat_images:
            viz_name = "|".join(self.input_keys + self.output_keys)
            viz_tensor = self.denorm(
                # concat by width
                torch.cat(input_tensors + output_tensors, dim=3)
            ).detach().cpu()
            visualizations[viz_name] = viz_tensor
        else:
            visualizations = dict(
                (k, self.denorm(v)) for k, v in zip(
                    self.input_keys + self.output_keys, input_tensors +
                    output_tensors
                )
            )
        return visualizations

    def save_visualizations(self, state, visualizations):
        tb_logger = self._get_tensorboard_logger(state)
        for key, batch_images in visualizations.items():
            batch_images = batch_images[:self.max_images]
            image = torchvision.utils.make_grid(batch_images, nrow=self._n_row)
            tb_logger.add_image(key, image, global_step=state.step)

    def visualize(self, state):
        visualizations = self.compute_visualizations(state)
        self.save_visualizations(state, visualizations)
        self._loader_visualized_in_current_epoch = True

    def on_loader_start(self, state: RunnerState):
        self._reset()

    def on_loader_end(self, state: RunnerState):
        if not self._loader_visualized_in_current_epoch:
            self.visualize(state)

    def on_batch_end(self, state: RunnerState):
        self._loader_batch_count += 1
        if self._loader_batch_count % self.batch_frequency:
            self.visualize(state)
