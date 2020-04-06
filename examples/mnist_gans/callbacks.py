# flake8: noqa
# isort: skip_file
import torch
import torchvision.utils

from catalyst.contrib.utils.tools.tensorboard import SummaryWriter
from catalyst.dl import Callback, CallbackOrder, State


class VisualizationCallback(Callback):
    TENSORBOARD_LOGGER_KEY = "_tensorboard"

    def __init__(
        self,
        input_keys=None,
        output_keys=None,
        batch_frequency=25,
        concat_images=True,
        max_images=20,
        num_rows=1,
        denorm="default",
    ):
        super().__init__(CallbackOrder.External)
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
            self.denorm = lambda x: x / 2 + 0.5
        elif denorm is None or denorm.lower() == "none":
            self.denorm = lambda x: x
        else:
            raise ValueError("unknown denorm fn")
        self._num_rows = num_rows
        self._reset()

    def _reset(self):
        self._loader_batch_count = 0
        self._loader_visualized_in_current_epoch = False

    @staticmethod
    def _get_tensorboard_logger(state: State) -> SummaryWriter:
        tb_key = VisualizationCallback.TENSORBOARD_LOGGER_KEY
        if (
            tb_key in state.callbacks
            and state.loader_name in state.callbacks[tb_key].loggers
        ):
            return state.callbacks[tb_key].loggers[state.loader_name]
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
            viz_tensor = (
                self.denorm(
                    # concat by width
                    torch.cat(input_tensors + output_tensors, dim=3)
                )
                .detach()
                .cpu()
            )
            visualizations[viz_name] = viz_tensor
        else:
            visualizations = dict(
                (k, self.denorm(v))
                for k, v in zip(
                    self.input_keys + self.output_keys,
                    input_tensors + output_tensors,
                )
            )
        return visualizations

    def save_visualizations(self, state: State, visualizations):
        tb_logger = self._get_tensorboard_logger(state)
        for key, batch_images in visualizations.items():
            batch_images = batch_images[: self.max_images]
            image = torchvision.utils.make_grid(
                batch_images, nrow=self._num_rows
            )
            tb_logger.add_image(key, image, global_step=state.global_step)

    def visualize(self, state):
        visualizations = self.compute_visualizations(state)
        self.save_visualizations(state, visualizations)
        self._loader_visualized_in_current_epoch = True

    def on_loader_start(self, state: State):
        self._reset()

    def on_loader_end(self, state: State):
        if not self._loader_visualized_in_current_epoch:
            self.visualize(state)

    def on_batch_end(self, state: State):
        self._loader_batch_count += 1
        if self._loader_batch_count % self.batch_frequency:
            self.visualize(state)


__all__ = ["VisualizationCallback"]
